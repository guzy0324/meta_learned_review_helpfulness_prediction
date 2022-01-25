from argparse import ArgumentParser
from json import loads
from os.path import exists
from typing import Any, Mapping, Sequence, Tuple, Union

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor, load as torch_load, tensor
from torch.nn import BCEWithLogitsLoss, LSTM, Linear, MSELoss, Module
from torch.nn.functional import relu, softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics import AUROC, Accuracy, Precision, Recall

from .configure_optimizers import configure_optimizers_func
from .datasets import Batch, DataModule, TripletBatch
from .embeddings import embedding
from .review_representations import review_representation_module
from .utils.add_args import add_args, add_args_func
from .utils.attention import mask_with_lengths
from .utils.file import LOGS
from .utils.switch import Switch

class PRH_Net(Module):
    def __init__(self, embed: str, embedding_args: Mapping[str, Any], num_layers: int, dropout: float):
        """
        Product-aware review helpfulness network.
        """
        super().__init__()
        self.embed = embedding(embed, **embedding_args)
        l = self.embed.embedding_dim
        self.output_dim = 2 * l
        self.Bi_LSTM_R = LSTM(l, l, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=dropout)
        self.Bi_LSTM_P = LSTM(l, l, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=dropout)
        self.W_P_b_P = Linear(2 * l, self.output_dim)

    def forward(self, review_batch: Tuple[Tensor, Sequence[int]], meta_batch: Tuple[Tensor, Sequence[int]]):
        R = self.embed(review_batch[0])  # (batch, n, l)
        P = self.embed(meta_batch[0])  # (batch, m, l)
        self.Bi_LSTM_R.flatten_parameters()
        H_R = pad_packed_sequence(self.Bi_LSTM_R(pack_padded_sequence(R, (n := review_batch[1]), batch_first=True))[0],
                                  batch_first=True)[0]  # (batch, n, 2l)
        self.Bi_LSTM_P.flatten_parameters()
        H_P = pad_packed_sequence(self.Bi_LSTM_P(pack_padded_sequence(P, (m := meta_batch[1]), batch_first=True))[0],
                                  batch_first=True)[0]  # (batch, m, 2l)
        Q = relu(self.W_P_b_P(H_P)) @ H_R.transpose(1, 2)  # (batch, m, n)
        mask_with_lengths(Q, m)
        G = softmax(Q, dim=1)  # (batch, m, n)
        H_R_hat = G.transpose(1, 2) @ H_P  # (batch, n, 2l)
        H = H_R + H_R_hat  # (batch, n, 2l)
        return H, n

@add_args_func("PRH_Net_module")
def PRH_Net_module_add_args(parser: ArgumentParser):
    add_args("embedding", parser)
    parser.add_argument("-nl", "--num_layers", type=int, default=2, help="num_layers for LSTM.")
    parser.add_argument("-do", "--dropout", type=int, default=0.2, help="dropout for LSTM.")

class PRH_Net_model(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.PRH_Net = PRH_Net(kwargs["embedding"], kwargs["embedding_args"], kwargs["num_layers"], kwargs["dropout"])
        self.review_representation = review_representation_module[kwargs["review_representation"]](
            self.PRH_Net.output_dim, **kwargs["review_representation_args"])
        self.linear = Linear(self.review_representation.output_dim, 1)
        self.review_representation_dim = self.review_representation.output_dim

    def forward(self, review_batch: Tuple[Tensor, Sequence[int]], meta_batch: Tuple[Tensor, Sequence[int]]):
        H, lengths = self.PRH_Net(review_batch, meta_batch)  # (batch, n, 2l)
        x, P = self.review_representation(H, lengths)  # (batch, output_dim)
        y_hat = self.linear(x).squeeze(1)  # (batch,)
        return y_hat, x, P

application_scenario_module = Switch()

@application_scenario_module("Identification")
class Identification(DataModule):
    def __init__(self, **kwargs):
        """
        Product-aware helpfulness prediction (identification) of online reviews.
        """
        super().__init__(**kwargs)
        if not exists(f"{LOGS}/vocab.tsv") or not exists(f"{LOGS}/pos_weight.txt"):
            self.prepare_data()
            self.setup("fit")
        if "pretrained_model" in kwargs and kwargs["pretrained_model"]:
            self.model = torch_load(f"{LOGS}/model.pth")
        else:
            self.model = PRH_Net_model(**kwargs)
        if self.hparams.auxiliary:
            if "pretrained_model" in kwargs and kwargs["pretrained_model"]:
                self.auxiliary_output_layer = torch_load(f"{LOGS}/auxiliary_output_layer.pth")
            else:
                self.auxiliary_output_layer = Linear(self.model.review_representation_dim, 1)
            self.auxiliary_loss = MSELoss(**self.hparams.auxiliary_args)
        with open(f"{LOGS}/pos_weight.txt") as f:
            pos_weight = tensor(float(f.read()))
        print(f"pos_weight = neg_num / pos_num = {pos_weight}")
        self.criterion = BCEWithLogitsLoss(pos_weight=pos_weight, **kwargs["criterion_args"])
        self.train_accuracy = Accuracy()
        self.train_precision = Precision()
        self.train_recall = Recall()
        self.val_auroc = AUROC()
        self.val_accuracy = Accuracy()
        self.val_precision = Precision()
        self.val_recall = Recall()
        self.test_auroc = AUROC()
        self.test_accuracy = Accuracy()
        self.test_precision = Precision()
        self.test_recall = Recall()

    def forward(self, review_batch: Tuple[Tensor, Sequence[int]], meta_batch: Tuple[Tensor, Sequence[int]]):
        return self.model(review_batch, meta_batch)

    def training_step(self, batch: Union[Batch, TripletBatch], *args, **kwargs) -> Tensor:
        y_hat, anchor, P = self(batch[0], batch[1])
        y = (batch[2] >= 0.75).int()
        batch_size = len(y)
        # logs losses for each training_step, and the average across the epoch, to the progress bar and logger
        loss = self.criterion(y_hat, y.float()) + P
        if self.hparams.triplet:
            _, positive, _ = self(batch[4], batch[5])
            _, negative, _ = self(batch[6], batch[7])
            triplet = self.hparams.triplet * self.triplet_loss(anchor, positive, negative)
            loss += triplet
            self.log("train_triplet", triplet, batch_size=batch_size)
        if self.hparams.auxiliary:
            auxiliary = self.hparams.auxiliary * self.auxiliary_loss(self.auxiliary_output_layer(anchor).squeeze(1), batch[3])
            loss += auxiliary
            self.log("train_auxiliary", auxiliary, batch_size=batch_size)
        self.log("train_P", P, batch_size=batch_size)
        self.log("train_loss", loss, batch_size=batch_size)
        y_hat = y_hat.sigmoid()
        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.train_accuracy(y_hat, y)
        self.log("train_accuracy", self.train_accuracy, batch_size=batch_size)
        self.train_precision(y_hat, y)
        self.log("train_precision", self.train_precision, batch_size=batch_size)
        self.train_recall(y_hat, y)
        self.log("train_recall", self.train_recall, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Union[Batch, TripletBatch], *args, **kwargs):
        y_hat, anchor, P = self(batch[0], batch[1])
        y = (batch[2] >= 0.75).int()
        batch_size = len(y)
        # logs losses at the end of the val_epoch, to logger
        loss = self.criterion(y_hat, y.float()) + P
        if self.hparams.triplet:
            _, positive, _ = self(batch[4], batch[5])
            _, negative, _ = self(batch[6], batch[7])
            triplet = self.hparams.triplet * self.triplet_loss(anchor, positive, negative)
            loss += triplet
            self.log("val_triplet", triplet, batch_size=batch_size)
        if self.hparams.auxiliary:
            auxiliary = self.hparams.auxiliary * self.auxiliary_loss(self.auxiliary_output_layer(anchor).squeeze(1), batch[3])
            loss += auxiliary
            self.log("val_auxiliary", auxiliary, batch_size=batch_size)
        self.log("val_P", P, batch_size=batch_size)
        self.log("val_loss", loss, batch_size=batch_size)
        y_hat = y_hat.sigmoid()
        # logs metrics at the end of the val_epoch, to logger
        self.val_auroc.update(y_hat, y)
        self.log("val_AUROC", self.val_auroc, batch_size=batch_size)
        self.val_accuracy(y_hat, y)
        self.log("val_accuracy", self.val_accuracy, batch_size=batch_size)
        self.val_precision(y_hat, y)
        self.log("val_precision", self.val_precision, batch_size=batch_size)
        self.val_recall(y_hat, y)
        self.log("val_recall", self.val_recall, batch_size=batch_size)

    def test_step(self, batch: Union[Batch, TripletBatch], *args, **kwargs):
        y_hat, anchor, P = self(batch[0], batch[1])
        y = (batch[2] >= 0.75).int()
        batch_size = len(y)
        # logs losses at the end of the test_epoch, to logger
        loss = self.criterion(y_hat, y.float()) + P
        if self.hparams.triplet:
            _, positive, _ = self(batch[4], batch[5])
            _, negative, _ = self(batch[6], batch[7])
            triplet = self.hparams.triplet * self.triplet_loss(anchor, positive, negative)
            loss += triplet
            self.log("test_triplet", triplet, batch_size=batch_size)
        if self.hparams.auxiliary:
            auxiliary = self.hparams.auxiliary * self.auxiliary_loss(self.auxiliary_output_layer(anchor).squeeze(1), batch[3])
            loss += auxiliary
            self.log("test_auxiliary", auxiliary, batch_size=batch_size)
        self.log("test_P", P, batch_size=batch_size)
        self.log("test_loss", loss, batch_size=batch_size)
        y_hat = y_hat.sigmoid()
        # logs metrics at the end of the test_epoch, to logger
        self.test_auroc.update(y_hat, y)
        self.log("test_AUROC", self.test_auroc, batch_size=batch_size)
        self.test_accuracy(y_hat, y)
        self.log("test_accuracy", self.test_accuracy, batch_size=batch_size)
        self.test_precision(y_hat, y)
        self.log("test_precision", self.test_precision, batch_size=batch_size)
        self.test_recall(y_hat, y)
        self.log("test_recall", self.test_recall, batch_size=batch_size)

    def configure_optimizers(self):
        return configure_optimizers_func[self.hparams.optimizer](self.parameters(), **self.hparams.optimizer_args)

    def configure_callbacks(self):
        # early_stop = EarlyStopping(monitor="val_loss")
        val_metric = "val_AUROC"
        checkpoint = ModelCheckpoint(monitor=val_metric,
                                     mode="max",
                                     filename=f"{{epoch}}-{{step}}-{{{val_metric}}}",
                                     save_top_k=8)
        # return [early_stop, checkpoint]
        return [checkpoint]

@add_args_func("PRH_Net")
def PRH_Net_add_args(parser: ArgumentParser, ignore_optimizer_args: bool = False):
    add_args("DataModule", parser)
    add_args("PRH_Net_module", parser)
    add_args("review_representation", parser)
    parser.add_argument("-a", "--auxiliary", type=float, default=0, help="Factor of auxiliary loss.")
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument("-aa", "--auxiliary_args", type=loads, default={}, help="Auxiliary loss arguments.")
    add_args("configure_optimizers", parser, ignore_optimizer_args=ignore_optimizer_args)
    parser.add_argument("-as",
                        "--application_scenario",
                        type=str,
                        choices=application_scenario_module.keys(),
                        required=True,
                        help="Application scenario.")
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument("-ca", "--criterion_args", type=loads, default={}, help="Criterion args.")
    parser.add_argument("-me", "--max_epochs", type=int, help="Maximum number of epochs.")
    parser.add_argument("-pm", "--pretrained_model", action="store_true", help="Use pretrained model.")
    parser.add_argument("-agb", "--accumulate_grad_batches", type=int, default=1, help="Accumulate grad batches.")

if __name__ == "__main__":
    from pickle import dump, load

    from matplotlib.pyplot import show
    from pytorch_lightning import Trainer
    from pytorch_lightning.tuner.tuning import Tuner

    from .utils.file import exc_remove
    from .utils.seed import seed

    parser = ArgumentParser(description="Product-aware helpfulness prediction of online reviews.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    fit = subparsers.add_parser("fit", help="Fit the model.")
    add_args("seed", fit)
    add_args("PRH_Net", fit)

    fit_from_checkpoint = subparsers.add_parser("fit_from_checkpoint", help="Fit the model from checkpoint.")
    fit_from_checkpoint.add_argument("-as",
                                     "--application_scenario",
                                     type=str,
                                     choices=application_scenario_module.keys(),
                                     required=True,
                                     help="Application scenario.")
    fit_from_checkpoint.add_argument("-p", "--path", type=str, required=True, help="Path to checkpoint.")

    lr_find = subparsers.add_parser("lr_find", help="Find learning rate.")
    add_args("seed", lr_find)
    add_args("PRH_Net", lr_find, ignore_optimizer_args=True)
    lr_find.add_argument("-lfa", "--lr_find-args", type=loads, default={}, help="lr_find arguments.")
    lr_find.add_argument("-r", "--redo", action="store_true", help="Redo.")

    test = subparsers.add_parser("test", help="Test the model.")
    add_args("seed", test)
    add_args("DataModule", test, test=True)
    test.add_argument("-as",
                      "--application_scenario",
                      type=str,
                      choices=application_scenario_module.keys(),
                      required=True,
                      help="Application scenario.")
    test.add_argument("-p", "--path", type=str, required=True, help="Path to checkpoint.")

    args = parser.parse_args()

    if args.stage == "fit":
        seed(args.seed)
        model = application_scenario_module[args.application_scenario](**args.__dict__)
        # 默认num_sanity_val_steps=2，有可能全是positive导致auroc报错
        trainer = Trainer(gpus=1,
                          default_root_dir=str(LOGS),
                          num_sanity_val_steps=0,
                          track_grad_norm=2,
                          detect_anomaly=True,
                          max_epochs=args.max_epochs,
                          accumulate_grad_batches=args.accumulate_grad_batches)
        trainer.fit(model)
    elif args.stage == "fit_from_checkpoint":
        model = application_scenario_module[args.application_scenario].load_from_checkpoint(path := args.path)
        assert args.application_scenario == model.hparams.application_scenario
        seed(model.hparams.seed)
        # 默认num_sanity_val_steps=2，有可能全是positive导致auroc报错
        trainer = Trainer(gpus=1,
                          default_root_dir=str(LOGS),
                          num_sanity_val_steps=0,
                          track_grad_norm=2,
                          detect_anomaly=True,
                          max_epochs=model.hparams.max_epochs,
                          accumulate_grad_batches=model.hparams.accumulate_grad_batches)
        trainer.fit(model, ckpt_path=path)
    elif args.stage == "lr_find":
        if args.redo or not exists(f"{LOGS}/lr_find_result.pkl"):
            seed(args.seed)
            model = application_scenario_module[args.application_scenario](optimizer_args={"lr": 0.001}, **args.__dict__)
            # 默认num_sanity_val_steps=2，有可能全是positive导致auroc报错
            trainer = Trainer(gpus=1, default_root_dir=str(LOGS), num_sanity_val_steps=0)
            tuner = Tuner(trainer)
            lr_finder = trainer.tuner.lr_find(model, **args.lr_find_args)
            fig = lr_finder.plot(suggest=True)
            try:
                # https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object
                with open(f"{LOGS}/lr_find_result.pkl", "wb") as f:
                    dump(fig, f)
            except:
                exc_remove(f"{LOGS}/lr_find_result.pkl")
        else:
            with open(f"{LOGS}/lr_find_result.pkl", "rb") as f:
                fig = load(f)
        show()
    elif args.stage == "test":
        seed(args.seed)
        model = application_scenario_module[args.application_scenario].load_from_checkpoint(path := args.path)
        assert args.application_scenario == model.hparams.application_scenario
        for arg in (args_dict := args.__dict__):
            setattr(model.hparams, arg, args_dict[arg])
        # 默认num_sanity_val_steps=2，有可能全是positive导致auroc报错
        trainer = Trainer(gpus=1,
                          default_root_dir=str(LOGS),
                          num_sanity_val_steps=0)
        trainer.test(model)
