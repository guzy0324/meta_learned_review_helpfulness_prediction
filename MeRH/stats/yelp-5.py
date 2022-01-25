if __name__ == "__main__":
    from os import mkdir
    from os.path import basename, splitext
    from shutil import rmtree
    from sys import argv

    from yaml import dump

    from ..utils.file import DATASET, json_load
    from ..prepare import yelp_5_keys

    path = splitext(basename(argv[0]))[0]
    rmtree(path, ignore_errors=True)
    mkdir(path)

    cat_none = []
    cat_unk = []
    cat_overlap = {}
    count = {key: 0 for key in list(yelp_5_keys) + ["none", "unk"]}

    for business in json_load(f"{DATASET}/yelp/yelp_academic_dataset_business.json"):
        if type(categories_str := business["categories"]) is not str:
            cat_none.append(f"{business['business_id']}\t{business['name']}\t{categories_str}")
            count["none"] += business["review_count"]
        else:
            if len(cat_in_yelp_5 := (set(categories_str.split(", ")) & yelp_5_keys)) == 1:
                count[list(cat_in_yelp_5)[0]] += business["review_count"]
            elif len(cat_in_yelp_5) == 0:
                cat_unk.append(f"{business['business_id']}\t{business['name']}\t{categories_str}")
                count["unk"] += business["review_count"]
            else:
                key = str(cat_in_yelp_5)
                if key not in cat_overlap:
                    cat_overlap[key] = []
                cat_overlap[key].append(f"{business['business_id']}\t{business['name']}\t{categories_str}")
                if key not in count:
                    count[key] = 0
                count[key] += business["review_count"]

    with open(f"{path}/yelp_cat_none.rc={count['none']}.stats", "w") as f:
        f.write("\n".join(cat_none))
    with open(f"{path}/yelp_cat_unk.rc={count['unk']}.stats", "w") as f:
        f.write("\n".join(cat_unk))
    for key in cat_overlap:
        with open(f"{path}/yelp_cat_overlap.{key}.rc={count[key]}.stats", "w") as f:
            f.write("\n".join(cat_overlap[key]))

    with open(f"{path}/count.stats", "w") as f:
        dump(count, f)
        f.write(f"yelp-5 reviews: {sum((v for k, v in count.items() if k in yelp_5_keys))}")
