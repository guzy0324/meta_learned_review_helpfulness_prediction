from argparse import ArgumentParser
from json import loads
from typing import Callable, Optional, Sequence, Tuple

from torch import Tensor, arange, eye, tensor
from torch.nn import Conv1d, LSTM, LayerNorm, Linear, Module, TransformerEncoder, TransformerEncoderLayer
from torch.nn.common_types import _size_1_t
from torch.nn.functional import pad, relu, softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .utils.add_args import add_args, add_args_func
from .utils.attention import lengths_to_mask, mask_with_lengths
from .utils.switch import Switch

review_representation_module = Switch()

@review_representation_module("HeadPlusTail")
class HeadPlusTail(Module):
    def __init__(self, embedding_dim: int):
        """
        Head + tail.
        """
        super().__init__()
        self.layer_norm = LayerNorm(embedding_dim)
        self.output_dim = embedding_dim

    def forward(self, e: Tensor, lengths: Sequence[int]) -> Tuple[Tensor, Tensor]:
        e_norm = self.layer_norm(e)
        x = (e_norm[:, 0, :] + e_norm[arange(len(lengths)), tensor(lengths) - 1, :]) / 2
        return x, 0

@review_representation_module("SelfAttention")
class SelfAttention(Module):
    def __init__(self, embedding_dim: int, r: int = 2, factor: float = 1):
        """
        self-attention
        """
        super().__init__()
        self.layer_norm = LayerNorm(embedding_dim)
        self.W_a = Linear(embedding_dim, embedding_dim, bias=False)
        self.r_T = Linear(embedding_dim, r, bias=False)
        self.factor = factor
        self.output_dim = r * embedding_dim

    def forward(self, e: Tensor, lengths: Sequence[int]) -> Tuple[Tensor, Tensor]:
        e_norm = self.layer_norm(e)
        attention_scores = self.r_T(self.W_a(e_norm).tanh())  # (batch, m, r)
        mask_with_lengths(attention_scores, lengths)
        a = softmax(attention_scores, dim=1)  # (batch, m, r)
        a_T = a.transpose(1, 2)  # (batch, r, m)
        x = (a_T @ e_norm).reshape(-1, self.output_dim)  # (batch, r * embedding_dim)
        r = a.shape[2]
        P = self.factor * ((a_T @ a - eye(r, r, device=a.device))**2).mean()  # (batch, r, r)
        return x, P

@review_representation_module("ABRR")
class ABRR(Module):
    def __init__(self, embedding_dim: int, k: _size_1_t, l: _size_1_t, output_layer: str = "SelfAttention", **kwargs):
        """
        Attention-based review representation in Multi-Task Neural Learning Architecture for End-to-End Identification of Helpful Reviews.
        """
        super().__init__()
        self.l = l
        self.layer_norm = LayerNorm(embedding_dim)
        self.conv = Conv1d(embedding_dim, k, l)
        self.output_layer = review_representation_module[output_layer](k, **kwargs)
        self.output_dim = self.output_layer.output_dim

    def forward(self, e: Tensor, lengths: Sequence[int]) -> Tuple[Tensor, Tensor]:
        if (m := e.shape[1]) < (l := self.l):
            e = pad(e, (0, 0, 0, l - m))
        Q = self.conv(self.layer_norm(e).transpose(1, 2)).tanh().transpose(1, 2)  # (batch, m - l + 1, k)
        return self.output_layer(Q, lengths)

@review_representation_module("BiLSTM")
class BiLSTM(Module):
    def __init__(self,
                 embedding_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_layer: str = "SelfAttention",
                 **kwargs):
        """
        BiLSTM.
        """
        super().__init__()
        self.layer_norm = LayerNorm(embedding_dim)
        self.bi_lstm = LSTM(embedding_dim,
                            embedding_dim,
                            batch_first=True,
                            bidirectional=True,
                            num_layers=num_layers,
                            dropout=dropout)
        self.output_layer = review_representation_module[output_layer](embedding_dim, **kwargs)
        self.embedding_dim = embedding_dim
        self.output_dim = self.output_layer.output_dim

    def forward(self, e: Tensor, lengths: Sequence[int]) -> Tuple[Tensor, Tensor]:
        self.bi_lstm.flatten_parameters()
        Q = pad_packed_sequence(self.bi_lstm(pack_padded_sequence(self.layer_norm(e), lengths, batch_first=True))[0],
                                batch_first=True)[0]  # (batch, m, 2 * embedding_dim)
        Q = Q[:, :, self.embedding_dim:] + Q[:, :, :self.embedding_dim]  # (batch, m, embedding_dim)
        return self.output_layer(e + Q, lengths)

@review_representation_module("Transformer")
class Transformer(Module):
    def __init__(self,
                 embedding_dim: int,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Callable[[Tensor], Tensor] = relu,
                 layer_norm_eps: float = 1e-6,
                 num_layers: int = 2,
                 norm: Optional[Module] = None,
                 output_layer: str = "SelfAttention",
                 **kwargs):
        """
        Transformer.
        """
        super().__init__()
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(embedding_dim,
                                    nhead,
                                    dim_feedforward,
                                    dropout,
                                    activation,
                                    layer_norm_eps,
                                    batch_first=True,
                                    norm_first=True), num_layers, norm)
        self.output_layer = review_representation_module[output_layer](embedding_dim, **kwargs)
        self.output_dim = self.output_layer.output_dim

    def forward(self, e: Tensor, lengths: Sequence[int]) -> Tuple[Tensor, Tensor]:
        Q = self.transformer_encoder(e, src_key_padding_mask=lengths_to_mask(lengths, e.shape[1],
                                                                             e.device))  # (batch, m, embedding_dim)
        return self.output_layer(e + Q, lengths)

@add_args_func("review_representation")
def review_representation_add_args(parser: ArgumentParser):
    parser.add_argument("-rr",
                        "--review_representation",
                        type=str,
                        choices=review_representation_module.keys(),
                        required=True,
                        help="Review representation.")
    parser.add_argument("-rra",
                        "--review_representation-args",
                        type=loads,
                        default={},
                        help="Attention-based review representation arguments.")

if __name__ == "__main__":
    from .datasets import DataModule
    from .embeddings import embedding
    from .utils.seed import seed

    parser = ArgumentParser(description="Review representations.")
    add_args("seed", parser)
    add_args("DataModule", parser)
    add_args("embedding", parser)
    add_args("review_representation", parser)
    args = parser.parse_args()

    seed(args.seed)
    datamodule = DataModule(**args.__dict__)
    datamodule.prepare_data()
    datamodule.setup()
    embed = embedding(args.embedding, dataloader=datamodule.train_dataloader(), **args.embedding_args)
    net = review_representation_module[args.review_representation](embed.embedding_dim, **args.review_representation_args)
    print(embed.embedding_dim)
    for review_batch, _, _ in datamodule.train_dataloader():
        print(review_batch)
        print(e := embed(review_batch[0]))
        print(e.shape)
        x, P = net(e, review_batch[1])
        print(x)
        print(x.shape)
        print(P)
        break
