if __name__ == "__main__":
    from json import dump
    from os import mkdir
    from os.path import basename, splitext
    from shutil import rmtree
    from sys import argv

    from ..utils.file import DATASET, LOGS, json_load
    from ..prepare import amazon_9_keys

    path = f"{LOGS}/{splitext(basename(argv[0]))[0]}"
    rmtree(path, ignore_errors=True)
    mkdir(path)

    for cat in amazon_9_keys:
        pos_neg = {}
        for review in json_load(f"{DATASET}/amazon-9/fit_{cat}.json"):
            if review[1] not in pos_neg:
                pos_neg[review[1]] = {"pos": [], "neg": []}
            if review[2] == 1:
                pos_neg[review[1]]["pos"].append(review[0])
            elif review[2] == 0:
                pos_neg[review[1]]["neg"].append(review[0])

        with open(f"{path}/{cat}.json", "w") as f:
            dump(pos_neg, f, indent=4)
