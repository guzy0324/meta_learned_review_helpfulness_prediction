from argparse import ArgumentParser
from typing import Optional

from pytorch_lightning import seed_everything
from torch.backends import cudnn

from .add_args import add_args, add_args_func

def seed(seed: Optional[int]):
    if seed is not None:
        # Seed everything
        seed_everything(seed)
        # Ensure that all operations are deterministic on GPU (if used) for reproducibility
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        print("Seed not set.")

@add_args_func("seed")
def seed_add_args(parser: ArgumentParser):
    parser.add_argument("-s", "--seed", type=int, help="Random seed.")

if __name__ == "__main__":
    from random import randint

    parser = ArgumentParser(description="Embeddings.")
    add_args("seed", parser)
    args = parser.parse_args()

    seed(args.seed)
    print([randint(1, 100) for _ in range(10)])
