from typing import Iterable

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as pad_sequence

def pad_sequence_ex(sequences: Iterable[Tensor], batch_first: bool = False, padding_value: float = 0.0):
    sequences = sorted(sequences, key=len, reverse=True)
    return pad_sequence(sequences, batch_first, padding_value), [len(v) for v in sequences]
