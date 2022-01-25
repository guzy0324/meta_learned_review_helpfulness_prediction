from math import log
from typing import Optional, Sequence

from torch import Tensor, arange, tensor, zeros
from torch.nn import Dropout, Module

def lengths_to_mask(lengths: Sequence[int], max_len: Optional[int] = None, device=None):
    # https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3
    return arange(max(lengths) if max_len is None else max_len, device=device)[None, :] >= tensor(lengths, device=device)[:, None]

def mask_with_lengths(input: Tensor, lengths: Sequence[int]):
    shape = input.shape
    dims = list(range(2, len(shape)))
    dims += (0, 1)
    mask = lengths_to_mask(lengths, shape[1], input.device)
    input.permute(dims).masked_fill_(mask, float("-inf"))

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = arange(max_len).unsqueeze(1)
        div_term = (arange(0, d_model, 2) * (-log(10000.0) / d_model)).exp()
        pe = zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = (position * div_term).sin()
        pe[:, 0, 1::2] = (position * div_term).cos()
        if batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] if not batch_first else [batch_size, seq_len, embedding_dim]
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            print(self.pe.shape)
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)

if __name__ == "__main__":
    from torch import zeros

    input = zeros(2, 3, 4)  # (batch, seq_len, n)
    lengths = [1, 2]
    mask_with_lengths(input, lengths)
    print(input)

    input = zeros(2, 3, 4)  # (batch, seq_len, n)
    pe = PositionalEncoding(4, batch_first=True)
    print(pe(input))
