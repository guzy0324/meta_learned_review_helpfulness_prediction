from argparse import ArgumentParser
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from json import dump, load, loads
from os import mkdir
from os.path import exists
from random import sample
from typing import AbstractSet, Any, Mapping, Sequence

# kaggle这样用没bug
import kaggle
from regex import match
from tqdm import tqdm

from .utils.add_args import add_args, add_args_func
from .utils.file import DATASET, download, exc_remove, json_load
from .utils.flatten import flatten
from .utils.switch import Switch

prepare_function = Switch()

def metadata_to_str(metadata: Mapping[str, Any], metadata_keys: AbstractSet[str]):
    return " ".join(flatten(metadata[key]) for key in metadata_keys if key in metadata)

def train_test_split(data: Sequence, train_percentage: float):
    train_indices = set(sample(range(len(data)), round(len(data) * train_percentage)))
    for i, d in enumerate(tqdm(data)):
        if i in train_indices:
            yield d, "fit"
        else:
            yield d, "test"

amazon_9_keys = {
    "Books", "Clothing, Shoes & Jewelry", "Electronics", "Grocery & Gourmet Food", "Health & Personal Care", "Home & Kitchen",
    "Movies & TV", "Pet Supplies", "Tools & Home Improvement"
}

def amazon_9_get_infix(cat: str) -> str:
    return cat.replace(",", "").replace(" ", "_").replace("&", "and")

@prepare_function("amazon")
def prepare_amazon(path: str):
    for cat in amazon_9_keys:
        for prefix in ("reviews", "meta"):
            download(
                f"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{prefix}_{amazon_9_get_infix(cat)}.json.gz",
                f"{path}")

amazon_9_metadata_keys = ["title", "brand", "categories", "description"]

def amazon_9_sub_process(path: str, train_percentage: float, cat: str):
    # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
    try:
        with open(f"{path}/asin_set.json") as f:
            asin_set = load(f)
        infix = amazon_9_get_infix(cat)
        meta_name = f"meta_{infix}.json.gz"
        metadata = OrderedDict(
            (d["asin"], [metadata_to_str(d, amazon_9_metadata_keys)])
            for d in json_load(f"{DATASET}/amazon/{meta_name}", std=False)
            if len(set(reduce(list.__add__, d["categories"]))
                   & amazon_9_keys) == 1 or (d["asin"] not in asin_set and asin_set.setdefault(d["asin"]) is None))
        print(f"Spliting {meta_name}...")
        for product, stage in train_test_split(metadata.keys(), train_percentage):
            metadata[product].append(stage)
        reviews_name = f"reviews_{infix}.json.gz"
        reviews = {"fit": [], "test": []}
        for d in json_load(f"{DATASET}/amazon/{reviews_name}", std=False):
            helpful, total = d["helpful"]
            if total > 0 and helpful <= total and not match("\s*$", reviewText := d["reviewText"]) and (asin :=
                                                                                                        d["asin"]) in metadata:
                product, stage = metadata[asin]
                reviews[stage].append([reviewText, product, helpful / total])
        for stage, data in reviews.items():
            fname = f"{stage}_{cat}.json"
            print(f"Writing {fname}...")
            with open(f"{path}/{fname}", "w") as f:
                for review in tqdm(data):
                    dump(review, f)
                    f.write("\n")
        with open(f"{path}/asin_set.json", "w") as f:
            dump(asin_set, f)
    except:
        return

@prepare_function("amazon-9")
def prepare_amazon_9(path: str, train_percentage: float = 0.8):
    if not exists(f"{DATASET}/amazon"):
        prepare("amazon")
    if exists(f"{path}/asin_set.json"):
        print("asin_set.json exists, skipping...")
    else:
        try:
            with open(f"{path}/asin_set.json", "w") as f:
                dump({}, f)
        except:
            exc_remove(f"{path}/asin_set.json")
    for cat in sorted(tuple(amazon_9_keys)):
        if (fit_path := exists(f"{path}/fit_{cat}.json")) and (test_path := exists(f"{path}/test_{cat}.json")):
            print(f"fit_{cat}.json and test_{cat}.json exists, skipping...")
        else:
            try:
                # https://stackoverflow.com/questions/15455048/releasing-memory-in-python
                with ProcessPoolExecutor(max_workers=1) as executor:
                    executor.submit(amazon_9_sub_process, path, train_percentage, cat).result()
            except:
                exc_remove(fit_path, test_path)

def amazon_9_splited_sub_process(path: str, cat: str):
    # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
    try:
        infix = amazon_9_get_infix(cat)
        meta_name = f"meta_{infix}.json.gz"
        metadata = OrderedDict((d["asin"], metadata_to_str(d, amazon_9_metadata_keys))
                               for d in json_load(f"{DATASET}/amazon/{meta_name}", std=False)
                               if len(set(reduce(list.__add__, d["categories"])) & amazon_9_keys) == 1)
        reviews_name = f"reviews_{infix}.json.gz"
        reviews = {}
        neg_pos = {}
        for d in json_load(f"{DATASET}/amazon/{reviews_name}", std=False):
            helpful, total = d["helpful"]
            if total > 0 and helpful <= total and not match("\s*$", reviewText := d["reviewText"]) and (asin :=
                                                                                                        d["asin"]) in metadata:
                if asin not in reviews:
                    reviews[asin] = []
                    neg_pos[asin] = [0, 0]
                reviews[asin].append([reviewText, metadata[asin], helpful / total, d["overall"]])
                neg_pos[asin][helpful / total >= 0.75] += 1
        folder_name = f"{path}/{cat}"
        mkdir(folder_name)
        print(f"Writing {folder_name}...")
        for asin, data in tqdm(reviews.items()):
            with open(f"{folder_name}/{len(data)}_{neg_pos[asin][0]}_{neg_pos[asin][1]}_{asin}.json", "w") as f:
                for review in data:
                    dump(review, f)
                    f.write("\n")
    except:
        return

@prepare_function("amazon-9_splited")
def prepare_amazon_9_splited(path: str):
    if not exists(f"{DATASET}/amazon"):
        prepare("amazon")
    for cat in sorted(tuple({"Grocery & Gourmet Food", "Pet Supplies"})):
        if folder_name := exists(f"{path}/{cat}"):
            print(f"{cat} exists, skipping...")
        else:
            try:
                # https://stackoverflow.com/questions/15455048/releasing-memory-in-python
                with ProcessPoolExecutor(max_workers=1) as executor:
                    executor.submit(amazon_9_splited_sub_process, path, cat).result()
            except:
                exc_remove(folder_name)

@prepare_function("yelp")
def prepare_yelp(path: str):
    if exists(f"{path}/Dataset_User_Agreement.pdf"):
        print("Yelp dataset exists, skipping...")
    else:
        try:
            kaggle.api.dataset_download_files("yelp-dataset/yelp-dataset", path=path, quiet=False, unzip=True)
        except:
            exc_remove(f"{path}/Dataset_User_Agreement.pdf")

yelp_5_keys = {"Beauty & Spas", "Health & Medical", "Home Services", "Restaurants", "Shopping"}
yelp_5_metadata_keys = ["name", "city", "categories", "attributes"]

# @prepare_function("yelp-5")
# 注意处理数据集存在，和ctrl-C的情况
# def prepare_yelp_5(path: str, train_percentage: float = 0.8):
#     if not exists(f"{DATASET}/yelp"):
#         prepare_func["yelp"]()
#     total = {cat: [] for cat in yelp_5_keys}
#     metadata = {key: [metadata_to_str(d, yelp_5_metadata_keys)] for d in json_load("{DATASET}/yelp/yelp_academic_dataset_business.json")
#                 if d["review_count"] > 0 and type(cat_str := d["categories"]) is str
#                 and len(cat_set := set(cat_str.split(", ")) & yelp_5_keys) == 1
#                 and total[min(cat_set)].append(key := d["business_id"]) is None}
#     print("Spliting metadata...")
#     for cat, businesses in total.items():
#         for business, stage in train_test_split(businesses, train_percentage):
#             metadata[business].append(stage)
#             metadata[business].append(cat)
#     reviews = {"fit": {cat: [] for cat in yelp_5_keys}, "test": {cat: [] for cat in yelp_5_keys}}
#     for d in json_load("{DATASET}/yelp/yelp_academic_dataset_review.json"):
#         if (total_vote := d["useful"] + d["funny"] + d["cool"]) > 0 and (key := d["business_id"]) in metadata:
#             business, stage, cat = metadata[key]
#             reviews[stage][cat].append([d["text"], business, d["useful"] / total_vote])
#     print("Saving...")
#     for stage, data in reviews.items():
#         for cat, reviews in data.items():
#             with open(f"{path}/{stage}_{cat}.json", "w") as f:
#                 for review in tqdm(reviews):
#                     dump(review, f)
#                     f.write("\n")

def prepare(dataset: str, **kwargs):
    path = f"{DATASET}/{dataset}"
    if not exists(path):
        mkdir(path)
    prepare_function[dataset](path=path, **kwargs)

@add_args_func("prepare")
def prepare_add_args(parser: ArgumentParser):
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument("-pa", "--prepare-args", type=loads, default={}, help="Preparing function arguments.")

if __name__ == "__main__":
    from .utils.seed import seed

    parser = ArgumentParser(description="Prepare dataset.")
    add_args("seed", parser)
    add_args("prepare", parser)
    parser.add_argument("-d", "--dataset", type=str, choices=prepare_function.keys(), required=True, help="Dataset to prepare.")
    args = parser.parse_args()

    seed(args.seed)
    prepare(args.dataset, **args.prepare_args)
