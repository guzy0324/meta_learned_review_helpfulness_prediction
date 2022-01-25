from json import loads
from os import mkdir
from os.path import exists

from fastNLP import Vocabulary
from fastNLP.embeddings import ElmoEmbedding, StaticEmbedding
from torch import Tensor
from torch.nn import Module

from .utils.add_args import add_args, add_args_func
from .utils.file import LOGS, EMBEDDINGS, download, unzip
from .utils.switch import Switch

embed_module = Switch()

@embed_module("word2vec")
class Word2Vec(Module):
    def __init__(self, path: str, **kwargs):
        super().__init__()
        download("http://download.fastnlp.top/embedding/GoogleNews-vectors-negative300.txt.gz", path)
        unzip(f"{path}/GoogleNews-vectors-negative300.txt.gz")
        vocab = Vocabulary.load(f"{LOGS}/vocab.tsv")
        print("Loading GoogleNews-vectors-negative300.txt...")
        self.embed = StaticEmbedding(vocab,
                                     model_dir_or_name=f"{path}/GoogleNews-vectors-negative300.txt",
                                     requires_grad=False,
                                     **kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.embedding_dim = self.embed.embedding_dim

    def forward(self, batch: Tensor) -> Tensor:
        return self.embed(batch)

@embed_module("glove")
class Glove(Module):
    def __init__(self, path: str, **kwargs):
        super().__init__()
        download("http://download.fastnlp.top/embedding/glove.840B.300d.zip", path)
        unzip(f"{path}/glove.840B.300d.zip")
        vocab = Vocabulary.load(f"{LOGS}/vocab.tsv")
        print("Loading glove.840B.300d...")
        self.embed = StaticEmbedding(vocab, model_dir_or_name=f"{path}/glove.840B.300d", **kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.embedding_dim = self.embed.embedding_dim

    def forward(self, batch: Tensor) -> Tensor:
        return self.embed(batch)

@embed_module("elmo")
class Elmo(Module):
    def __init__(self, path: str, **kwargs):
        super().__init__()
        download("http://download.fastnlp.top/embedding/elmo_en_Medium.zip", path)
        unzip(f"{path}/elmo_en_Medium.zip")
        vocab = Vocabulary.load(f"{LOGS}/vocab.tsv")
        print("Loading elmo_en_Medium.zip...")
        self.embed = ElmoEmbedding(vocab, model_dir_or_name=f"{path}/elmo_en_Medium", **kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.embedding_dim = self.embed.embedding_dim

    def forward(self, batch: Tensor) -> Tensor:
        return self.embed(batch)

def embedding(embed: str, **kwargs) -> Module:
    path = f"{EMBEDDINGS}/{embed}"
    if not exists(path):
        mkdir(path)
    return embed_module[embed](path=path, **kwargs)

@add_args_func("embedding")
def embedding_add_args(parser):
    parser.add_argument("-e", "--embedding", type=str, choices=embed_module.keys(), required=True, help="Embedding.")
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument("-ea", "--embedding-args", type=loads, default={}, help="Embedding arguments.")

if __name__ == "__main__":
    from argparse import ArgumentParser

    from .datasets_reproduce import DataModule
    from .utils.seed import seed

    parser = ArgumentParser(description="Embeddings.")
    add_args("seed", parser)
    add_args("DataModule", parser)
    add_args("embedding", parser)
    args = parser.parse_args()

    seed(args.seed)
    datamodule = DataModule(**args.__dict__)
    datamodule.prepare_data()
    datamodule.setup()
    embed = embedding(args.embedding, **args.embedding_args)
    print(list(embed.named_parameters()))
    print(embed.embedding_dim)
    for review_batch, _, _ in datamodule.train_dataloader():
        print(review_batch)
        print([[embed.vocab.to_index(word) for word in sentence] for sentence in review_batch[0]])
        print(out := embed(review_batch[0]))
        print(out.shape)
        print((out != 0).sum())
        break
