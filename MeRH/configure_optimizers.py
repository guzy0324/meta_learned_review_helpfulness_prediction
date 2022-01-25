from argparse import ArgumentParser
from json import loads
from typing import Iterable, Union

from torch import Tensor
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CyclicLR

from .utils.add_args import add_args, add_args_func
from .utils.switch import Switch

configure_optimizers_func = Switch({"AdamW": AdamW})

@configure_optimizers_func("Nesterov")
def configure_Nesterov(params: Iterable[Union[Tensor, dict]], lr: float):
    return SGD(params, lr, momentum=0.8, nesterov=True)

@configure_optimizers_func("Nesterov with CLR")
def configure_Nesterov_with_CLR(params: Iterable[Union[Tensor, dict]], base_lr: float, max_lr: float, step_size_up: int):
    optimizer = SGD(params, base_lr, momentum=0.8, nesterov=True)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": CyclicLR(optimizer, base_lr, max_lr, step_size_up)
        },
    }

@add_args_func("configure_optimizers")
def configure_optimizers_add_args(parser: ArgumentParser, ignore_optimizer_args=False):
    parser.add_argument("-o", "--optimizer", type=str, required=True, help="Optimizer.")
    if not ignore_optimizer_args:
        parser.add_argument("-oa", "--optimizer-args", type=loads, default={}, help="Optimizer arguments.")

if __name__ == "__main__":
    from torch.nn import Linear

    parser = ArgumentParser(description="Optimizers.")
    add_args("configure_optimizers", parser)
    args = parser.parse_args()
    print(configure_optimizers_func[args.optimizer](Linear(1, 1).parameters(), **args.optimizer_args))
