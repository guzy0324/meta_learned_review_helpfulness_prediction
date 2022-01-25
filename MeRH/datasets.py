from argparse import ArgumentParser
from functools import reduce
from json import loads
from os import listdir
from os.path import exists
from random import choice, sample, shuffle
from typing import Iterable, Optional, List, Sequence, SupportsIndex, Tuple, Union

from fastNLP import Vocabulary
from pytorch_lightning import LightningModule
from torch import Tensor, LongTensor, stack, tensor
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

from .prepare import amazon_9_keys, prepare, yelp_5_keys
from .utils.add_args import add_args, add_args_func
from .utils.file import DATASET, LOGS, exc_remove, json_load
from .utils.hidden_prints import HiddenPrints
from .utils.preprocess import clean_str
from .utils.rnn import pad_sequence_ex
from .utils.switch import Switch

DatasetItem = Tuple[Union[List[str], Tensor], Union[List[str], Tensor], Tensor, Tensor]

def vocab_fit(reviews: Iterable[DatasetItem]):
    if exists(f"{LOGS}/vocab.tsv"):
        print("Vocabulary exists, loading vocabulary...")
        vocab = Vocabulary.load(f"{LOGS}/vocab.tsv")
    else:
        print("Building vocabulary...")
        vocab = Vocabulary()
        for review, metadata, _, _ in tqdm(reviews):
            vocab.add_word_lst(review)
            vocab.add_word_lst(metadata)
        try:
            vocab.save(f"{LOGS}/vocab.tsv")
        except:
            exc_remove(f"{LOGS}/vocab.tsv")

def vocab_transform(reviews: Sequence[DatasetItem]):
    vocab = Vocabulary.load(f"{LOGS}/vocab.tsv")
    print("Word2idx...")
    for i in tqdm(range(len(reviews))):
        review, metadata, score, rating = reviews[i]
        reviews[i] = (LongTensor([vocab.to_index(word) for word in review]),
                      LongTensor([vocab.to_index(word) for word in metadata]), tensor(score), tensor(rating))

dataset_class = Switch()

@dataset_class("amazon-9_splited")
class Amazon9Dataset(Dataset):
    def __init__(self, subset: Sequence[str], max_len: int, count: int, num: int = -1):
        super().__init__()
        tokenizer = get_tokenizer("spacy")
        self.reviews = []
        for cat in sorted(tuple(amazon_9_keys if subset is None else amazon_9_keys & set(subset))):
            print(f"Loading {(folder_name := f'{DATASET}/amazon-9_splited/{cat}')}...")
            for fname in tqdm(sorted(listdir(folder_name), key=lambda x: int(x.split("_")[0]))):
                if num == 0:
                    return
                if int(fname.split("_")[0]) >= count:
                    with HiddenPrints():
                        self.reviews += ((tokenizer(clean_str(review))[:max_len], tokenizer(clean_str(metadata))[:max_len], score,
                                          rating) for review, metadata, score, rating in json_load(f"{folder_name}/{fname}"))
                        num -= 1

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx: SupportsIndex) -> DatasetItem:
        return self.reviews[idx]

DatasetForTripletItem = Tuple[Union[str, Tensor], Union[str, Tensor], Tensor, Union[str, Tensor], Union[str, Tensor],
                              Union[str, Tensor], Union[str, Tensor]]

class DatasetForTriplet(Dataset):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset
        self.splited = [[], []]
        for i, (_, _, score, _) in enumerate(DataLoader(self.dataset)):
            self.splited[score >= 0.75].append(i)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: SupportsIndex) -> DatasetForTripletItem:
        review, metadata, score, rating = self.dataset[idx]
        label = score >= 0.75
        while (pos_idx := choice(self.splited[label])) == idx:
            pass
        pos_review, pos_metadata, _, _ = self.dataset[pos_idx]
        neg_review, neg_metadata, _, _ = self.dataset[choice(self.splited[not label])]
        return review, metadata, score, rating, pos_review, pos_metadata, neg_review, neg_metadata

Batch = Tuple[Tuple[Tensor, List[int]], Tuple[Tensor, List[int]], Tensor, Tensor]

def collate_fn(batch: Sequence[DatasetItem]) -> Batch:
    review_batch, metadata_batch, score_batch, rating_batch = zip(*batch)
    return pad_sequence_ex(review_batch, True), pad_sequence_ex(metadata_batch, True), stack(score_batch), stack(rating_batch)

TripletBatch = Tuple[Tuple[Tensor, List[int]], Tuple[Tensor, List[int]], Tensor, Tensor,
                     Tuple[Tuple[Tensor, List[int]], Tuple[Tensor, List[int]], Tuple[Tuple[Tensor, List[int]], Tuple[Tensor,
                                                                                                                     List[int]]]]]

def collate_fn_for_triplet(batch: Sequence[DatasetForTripletItem]) -> TripletBatch:
    review_batch, metadata_batch, score_batch, rating_batch, pos_review_batch, pos_metadata_batch, neg_review_batch, neg_metadata_batch = zip(
        *batch)
    return pad_sequence_ex(review_batch,
                           True), pad_sequence_ex(metadata_batch, True), stack(score_batch), rating_batch, pad_sequence_ex(
                               pos_review_batch,
                               True), pad_sequence_ex(pos_metadata_batch,
                                                      True), pad_sequence_ex(neg_review_batch,
                                                                             True), pad_sequence_ex(neg_metadata_batch, True)

class DataModule(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        prepare(self.hparams.dataset, **self.hparams.prepare_args)

    def setup(self, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full = dataset_class[self.hparams.dataset](self.hparams.subset, self.hparams.max_len, **self.hparams.dataset_args)
            len_val = int(round(len(full) * self.hparams.val_percentage))
            self.train_set, self.val_set = random_split(full, [len(full) - len_val, len_val])
            vocab_fit(self.train_set[i] for i in tqdm(range(len(self.train_set))))
            vocab_transform(full.reviews)
            if not exists(f"{LOGS}/pos_weight.txt"):
                print("Calculating pos_weight...")
                pos_num = sum(
                    float((batch_tuple[2] >= 0.75).sum()) for batch_tuple in tqdm(
                        DataLoader(self.train_set,
                                   batch_size=self.hparams.batch_size,
                                   num_workers=self.hparams.num_workers,
                                   pin_memory=True,
                                   collate_fn=collate_fn)))
                try:
                    with open(f"{LOGS}/pos_weight.txt", "w") as f:
                        f.write(str((len(self.train_set) - pos_num) / pos_num))
                except:
                    exc_remove(f"{LOGS}/pos_weight.txt")
        elif stage == "test":
            self.test_set = dataset_class[self.hparams.dataset](self.hparams.subset, self.hparams.max_len,
                                                                **self.hparams.dataset_args)
            vocab_transform(self.test_set.reviews)
        if self.hparams.triplet:
            self.triplet_loss = TripletMarginLoss(**self.hparams.triplet_args)
            if stage == "fit" or stage is None:
                self.train_set = DatasetForTriplet(self.train_set)
                self.val_set = DatasetForTriplet(self.val_set)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn_for_triplet if self.hparams.triplet else collate_fn,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn_for_triplet if self.hparams.triplet else collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn_for_triplet if self.hparams.triplet else collate_fn)

@add_args_func("DataModule")
def DataModule_add_args(parser: ArgumentParser, ignore_batch_size: bool = False, test: bool = False):
    add_args("prepare", parser)
    parser.add_argument("-d", "--dataset", type=str, choices=dataset_class.keys(), required=True, help="Dataset to prepare.")
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument("-da", "--dataset_args", type=loads, default={}, help="Dataset arguments.")
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument("-ss", "--subset", type=loads, help="Subset of data to use.")
    parser.add_argument("-ml", "--max_len", type=int, default=1024, help="Max length of sequence.")
    if not test:
        parser.add_argument("-vp", "--val-percentage", type=float, default=0.25, help="Validation percentage.")
        parser.add_argument("-t", "--triplet", type=float, default=0, help="Factor of triplet loss.")
        # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
        parser.add_argument("-ta", "--triplet_args", type=loads, default={}, help="TripletMarginLoss arguments.")
    if not ignore_batch_size:
        parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("-nw", "--num-workers", type=int, default=1, help="Number of workers.")

def sample_support_or_query(support_or_query: Sequence[DatasetItem], k_shot: int, inner_batch_size: int):
    k_shot = min(k_shot, len(support_or_query))
    shuffle(support_or_query)
    left = 0
    return [
        collate_fn(support_or_query[left:(left := min(right, k_shot))])
        for right in range(inner_batch_size, k_shot + inner_batch_size, inner_batch_size)
    ]

dataset_for_meta_learning_class = Switch()

@dataset_for_meta_learning_class("amazon-9_splited")
class Amazon9SplitedDatasetForMetaLearning(Dataset):
    def __init__(self,
                 subset: Sequence[str],
                 max_len: int,
                 k_shot: int = 8,
                 inner_batch_size: int = 2,
                 min_count: int = 20,
                 max_count: int = 100,
                 sample_count: int = 2000):
        super().__init__()
        tokenizer = get_tokenizer("spacy")
        fnames = []
        for cat in sorted(tuple(amazon_9_keys if subset is None else amazon_9_keys & set(subset))):
            folder_name = f"{DATASET}/amazon-9_splited/{cat}"
            fnames += (fname for fname in listdir(folder_name) if min_count <= (count := int((splited := fname.split("_"))[0]))
                       and count < max_count and int(splited[1]) > 1 and int(splited[2]) > 1)
        reviews = []
        print(f"{(sample_count := min(sample_count, len(fnames)))} products are sampled")
        for fname in tqdm(sample(fnames, sample_count)):
            reviews.append(([], []))
            with HiddenPrints():
                for review, metadata, score, rating in json_load(f"{folder_name}/{fname}"):
                    reviews[-1][score >= 0.75].append(
                        (tokenizer(clean_str(review))[:max_len], tokenizer(clean_str(metadata))[:max_len], score, rating))
        self.support = []
        self.query = []
        print("Spliting data into support set and query set...")
        for reviews_per_product in tqdm(reviews):
            shuffle(reviews_per_product[0])
            shuffle(reviews_per_product[1])
            self.support.append(reviews_per_product[0][(mid0 := min(len(reviews_per_product[0]) // 2, k_shot)):] +
                                reviews_per_product[1][(mid1 := min(len(reviews_per_product[1]) // 2, k_shot)):])
            self.query.append(reviews_per_product[0][:mid0] + reviews_per_product[1][:mid1])
        self.k_shot = k_shot
        self.inner_batch_size = inner_batch_size

    def __len__(self):
        return len(self.support)

    def __getitem__(self, idx: SupportsIndex):
        return sample_support_or_query(self.support[idx], self.k_shot,
                                       self.inner_batch_size), sample_support_or_query(self.query[idx], self.k_shot,
                                                                                       self.inner_batch_size)

def collate_fn_for_meta_learning(batch: Sequence[Tuple[Batch, Batch]]):
    return batch

class DataModuleForMetaLearning(DataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        loaded = False
        if not exists(f"{LOGS}/vocab.tsv") or not exists(f"{LOGS}/pos_weight.txt"):
            full = dataset_for_meta_learning_class[self.hparams.dataset](self.hparams.subset, self.hparams.max_len,
                                                                         **self.hparams.dataset_args)
            print("Catenating full_support...")
            full_support = reduce(list.__add__, tqdm(full.support))
            loaded = True
        if not exists(f"{LOGS}/vocab.tsv"):
            vocab_fit(full_support)
        if not exists(f"{LOGS}/pos_weight.txt"):
            print("Calculating pos_weight...")
            pos_num = float(sum(support[2] >= 0.75 for support in tqdm(full_support)))
            try:
                with open(f"{LOGS}/pos_weight.txt", "w") as f:
                    f.write(str((len(full_support) - pos_num) / pos_num))
            except:
                exc_remove(f"{LOGS}/pos_weight.tsv")
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if not loaded:
                full = dataset_for_meta_learning_class[self.hparams.dataset](self.hparams.subset, self.hparams.max_len,
                                                                             **self.hparams.dataset_args)
            print("word2idx for full.support...")
            for support in tqdm(full.support):
                with HiddenPrints():
                    vocab_transform(support)
            print("word2idx for full.query...")
            for query in tqdm(full.query):
                with HiddenPrints():
                    vocab_transform(query)
            len_val = int(round(len(full) * self.hparams.val_percentage))
            self.train_set, self.val_set = random_split(full, [len(full) - len_val, len_val])

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn_for_meta_learning,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn_for_meta_learning)

@add_args_func("DataModuleForMetaLearning")
def DataModuleForMetaLearning_add_args(parser: ArgumentParser):
    add_args("prepare", parser)
    parser.add_argument("-bs", "--batch-size", type=int, required=True, help="Task batch size.")
    parser.add_argument("-d",
                        "--dataset",
                        type=str,
                        choices=dataset_for_meta_learning_class.keys(),
                        required=True,
                        help="Dataset to prepare.")
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument("-da", "--dataset_args", type=loads, default={}, help="Dataset arguments.")
    parser.add_argument("-ml", "--max_len", type=int, default=1024, help="Max length of sequence.")
    parser.add_argument("-nw", "--num-workers", type=int, default=1, help="Number of workers.")
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument("-ss", "--subset", type=loads, help="Subset of data to use.")
    parser.add_argument("-vp", "--val-percentage", type=float, default=0.25, help="Validation percentage.")

if __name__ == "__main__":
    from .utils.seed import seed

    parser = ArgumentParser(description="Datasets.")
    subparsers = parser.add_subparsers(dest="type", required=True)
    data_module = subparsers.add_parser("data_module", help="Data module.")
    add_args("seed", data_module)
    add_args("DataModule", data_module)
    data_module_for_meta_learning = subparsers.add_parser("data_module_for_meta_learning", help="Data module for meta learning.")
    add_args("seed", data_module_for_meta_learning)
    add_args("DataModuleForMetaLearning", data_module_for_meta_learning)
    args = parser.parse_args()

    if args.type == "data_module":
        seed(args.seed)
        datamodule = DataModule(**args.__dict__)
        datamodule.prepare_data()
        datamodule.setup()
        for i in datamodule.train_dataloader():
            print(i)
            break
        print(len(datamodule.train_dataloader()))
        print(len(datamodule.val_dataloader()))
        print(sum((score_batch <= 0.75).sum() for _, _, score_batch, _ in tqdm(datamodule.train_dataloader())))
        print(sum((score_batch <= 0.75).sum() for _, _, score_batch, _ in tqdm(datamodule.val_dataloader())))
        datamodule.teardown()
    elif args.type == "data_module_for_meta_learning":
        seed(args.seed)
        datamodule = DataModuleForMetaLearning(**args.__dict__)
        datamodule.prepare_data()
        datamodule.setup()
        print(len(datamodule.train_dataloader()))
        print(len(datamodule.val_dataloader()))
        batch = next(iter(datamodule.train_dataloader()))
        print("batch", len(batch))
        print("batch[0]", len(batch[0]))
        print(batch[0])
        print("batch[0][0]", len(batch[0][0]))
        print(batch[0][0])
        datamodule.teardown()
