if __name__ == "__main__":
    from os import mkdir
    from os.path import basename, splitext
    from shutil import rmtree
    from sys import argv

    from matplotlib.pyplot import subplots

    from ..utils.file import DATASET, LOGS, json_load
    from ..prepare import amazon_9_keys

    path = f"{LOGS}/{splitext(basename(argv[0]))[0]}"
    rmtree(path, ignore_errors=True)
    mkdir(path)

    interval = 0.05

    for cat in amazon_9_keys:
        score_stats = [0] * (len_stats := int(1 / interval + 1))
        for review in json_load(f"{DATASET}/amazon-9/fit_{cat}.json"):
            i = min(int(review[2] // interval), len_stats - 1)
            score_stats[i] += 1
        for review in json_load(f"{DATASET}/amazon-9/test_{cat}.json"):
            i = min(int(review[2] // interval), len_stats - 1)
            score_stats[i] += 1

        fig, ax = subplots(figsize=(20, 14))
        ax.bar([f"{i * interval:.2f}" + (f"-{(i + 1) * interval:.2f}" if i + 1 < len_stats else "+") for i in range(len_stats)],
               score_stats)
        fig.savefig(f"{path}/{cat}_{sum(score_stats[int(0.75 / interval):])/sum(score_stats)}.png")
