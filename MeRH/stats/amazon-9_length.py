if __name__ == "__main__":
    from os import mkdir
    from os.path import basename, splitext
    from shutil import rmtree
    from sys import argv

    from matplotlib.pyplot import subplots, title, xlabel, ylabel
    from torchtext.data.utils import get_tokenizer

    from ..utils.file import DATASET, LOGS, json_load
    from ..prepare import amazon_9_keys

    path = f"{LOGS}/{splitext(basename(argv[0]))[0]}"
    rmtree(path, ignore_errors=True)
    mkdir(path)

    interval = 100
    max_len = 2000

    tokenizer = get_tokenizer("spacy")
    for cat in amazon_9_keys:
        review_length_stats = []
        meta_length_stats = []
        for review in json_load(f"{DATASET}/amazon-9/fit_{cat}.json"):
            length = len(tokenizer(review[0]))
            i = length // interval
            while i >= len(review_length_stats):
                review_length_stats.append(0)
            review_length_stats[i] += 1

            length = len(tokenizer(review[1]))
            i = length // interval
            while i >= len(meta_length_stats):
                meta_length_stats.append(0)
            meta_length_stats[i] += 1

        for review in json_load(f"{DATASET}/amazon-9/test_{cat}.json"):
            length = len(tokenizer(review[0]))
            i = length // interval
            while i >= len(review_length_stats):
                review_length_stats.append(0)
            review_length_stats[i] += 1

            length = len(tokenizer(review[1]))
            i = length // interval
            while i >= len(meta_length_stats):
                meta_length_stats.append(0)
            meta_length_stats[i] += 1

        max_len_stats = max_len // interval + 1
        if max_len_stats < len(review_length_stats):
            review_length_stats[max_len_stats - 1] += sum(review_length_stats[max_len_stats:])
            review_length_stats = review_length_stats[:max_len_stats]
        if max_len_stats < len(meta_length_stats):
            meta_length_stats[max_len_stats - 1] += sum(meta_length_stats[max_len_stats:])
            meta_length_stats = meta_length_stats[:max_len_stats]

        fig, ax = subplots(figsize=(12, 4))
        bar = ax.bar([f"{i:d}-{(i + 1):d}" if i < max_len_stats - 1 else f"{i:d}+" for i in range(len(review_length_stats))],
                     review_length_stats)
        for b in bar:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2., h, f"{h:d}", ha="center", va="bottom")
        title(f"{cat} review", y=1, loc="right")
        xlabel(f"length of reivew / {interval}")
        ylabel(f"count")
        fig.savefig(f"{path}/{cat}_review.png")

        fig, ax = subplots(figsize=(12, 4))
        bar = ax.bar([f"{i:d}-{(i + 1):d}" if i < max_len_stats - 1 else f"{i:d}+" for i in range(len(meta_length_stats))],
                     meta_length_stats)
        for b in bar:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2., h, f"{h:d}", ha="center", va="bottom")
        title(f"{cat} metadata", y=1, loc="right")
        xlabel(f"length of metadata / {interval}")
        ylabel(f"count")
        fig.savefig(f"{path}/{cat}_meta.png")
