if __name__ == "__main__":
    from functools import reduce
    from os import mkdir
    from os.path import basename, splitext
    from shutil import rmtree
    from sys import argv

    from yaml import dump
    from regex import match

    from ..utils.file import DATASET, LOGS, json_load
    from ..prepare import amazon_9_keys, amazon_9_get_infix

    path = f"{LOGS}/{splitext(basename(argv[0]))[0]}"
    rmtree(path, ignore_errors=True)
    mkdir(path)

    core = 5
    range_core = range(core)

    for cat in amazon_9_keys:
        infix = amazon_9_get_infix(cat)

        invalid_count = 0
        review_count = {}
        review_count_1v = {}
        review_count_75hr = {}
        for review in json_load(f"{DATASET}/amazon/reviews_{infix}.json.gz", std=False):
            helpful, total = review["helpful"]
            if match("\s*$", review["reviewText"]) or helpful > total:
                invalid_count += 1
            else:
                if (asin := review["asin"]) not in review_count:
                    review_count[asin] = 0
                    review_count_1v[asin] = 0
                    review_count_75hr[asin] = 0
                review_count[asin] += 1
                if total > 0:
                    review_count_1v[asin] += 1
                    if helpful / total >= 0.75:
                        review_count_75hr[asin] += 1

        print(f"Loading meta_{infix}.json.gz...")
        amazon_keys = (list_amazon_9_keys := list(amazon_9_keys)) + ["none", "unk", "few reviews"]
        count_reviews = {i: {key: 0 for key in amazon_keys} for i in range_core}
        count_reviews_1v = {i: {key: 0 for key in list_amazon_9_keys} for i in range_core}
        count_reviews_75hr = {i: {key: 0 for key in list_amazon_9_keys} for i in range_core}
        count_products = {i: {key: 0 for key in amazon_keys} for i in range_core}
        for product in json_load(f"{DATASET}/amazon/meta_{infix}.json.gz", std=False):
            if (asin := product["asin"]) in review_count:
                for i in range(core):
                    count_reviews_core = count_reviews[i]
                    count_reviews_1v_core = count_reviews_1v[i]
                    count_reviews_75hr_core = count_reviews_75hr[i]
                    count_products_core = count_products[i]
                    if (rc := review_count[asin]) > i:
                        if "categories" not in product:
                            count_reviews_core["none"] += rc
                            count_products_core["none"] += 1
                        elif len(cats_in_amazon_9 := set(reduce(list.__add__, product["categories"])) & amazon_9_keys) == 1:
                            cat_in_amazon_9 = list(cats_in_amazon_9)[0]
                            count_reviews_core[cat_in_amazon_9] += rc
                            count_reviews_1v_core[cat_in_amazon_9] += review_count_1v[asin]
                            count_reviews_75hr_core[cat_in_amazon_9] += review_count_75hr[asin]
                            count_products_core[cat_in_amazon_9] += 1
                        elif len(cats_in_amazon_9) == 0:
                            count_reviews_core["unk"] += rc
                            count_products_core["unk"] += 1
                        else:
                            if (key := str(cats_in_amazon_9)) not in count_reviews_core:
                                count_reviews_core[key] = 0
                                count_reviews_1v_core[key] = 0
                                count_reviews_75hr_core[key] = 0
                            count_reviews_core[key] += rc
                            count_reviews_1v_core[key] += review_count_1v[asin]
                            count_reviews_75hr_core[key] += review_count_75hr[asin]
                            if key not in count_products_core:
                                count_products_core[key] = 0
                            count_products_core[key] += 1
                    else:
                        count_reviews_core["few reviews"] += rc
                        count_products_core["few reviews"] += 1

        for i in range(0, core):
            count_reviews_core = count_reviews[i]
            count_reviews_1v_core = count_reviews_1v[i]
            count_reviews_75hr_core = count_reviews_75hr[i]
            count_products_core = count_products[i]
            c = i + 1
            with open(f"{path}/{infix}_{c}-core_reviews.stats", "w") as f:
                dump(count_reviews_core, f)
                f.write(f"total: {sum(count_reviews_core.values())}\n")
                f.write(f"invalid total: {invalid_count}\n")
                f.write(f"\n1v\n")
                dump(count_reviews_1v_core, f)
                f.write(f"total: {sum(count_reviews_1v_core.values())}\n")
                f.write(f"\n75hr\n")
                dump(count_reviews_75hr_core, f)
                f.write(f"total: {sum(count_reviews_75hr_core.values())}")
            with open(f"{path}/{infix}_{c}-core_products.stats", "w") as f:
                dump(count_products_core, f)
                f.write(f"total: {sum(count_products_core.values())}")
