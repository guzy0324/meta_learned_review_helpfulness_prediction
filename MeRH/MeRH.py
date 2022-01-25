from argparse import ArgumentParser
from copy import deepcopy
from json import loads
from os.path import exists
from typing import Sequence, Tuple

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor, cat, set_grad_enabled, save, tensor
from torch.nn import BCEWithLogitsLoss, Linear, MSELoss
from torchmetrics import AUROC, Accuracy, Precision, Recall

from .PRH_Net import PRH_Net_model
from .configure_optimizers import configure_optimizers_func
from .datasets import Batch, DataModuleForMetaLearning
from .utils.add_args import add_args, add_args_func
from .utils.file import LOGS
from .utils.switch import Switch

application_scenario_module = Switch()

@application_scenario_module("Identification")
class Identification(DataModuleForMetaLearning):
    def __init__(self, **kwargs):
        """
        Meta-learned review helpfulness prediction (identification).
        """
        super().__init__(**kwargs)
        if not exists(f"{LOGS}/vocab.tsv") or not exists(f"{LOGS}/pos_weight.txt"):
            self.prepare_data()
            self.setup("fit")
        self.model = PRH_Net_model(**kwargs)
        if self.hparams.auxiliary:
            self.auxiliary_output_layer = Linear(self.model.review_representation_dim, 1)
            self.auxiliary_loss = MSELoss(**self.hparams.auxiliary_args)
        with open(f"{LOGS}/pos_weight.txt") as f:
            pos_weight = tensor(float(f.read()))
        print(f"pos_weight = neg_num / pos_num = {pos_weight}")
        self.criterion = BCEWithLogitsLoss(pos_weight=pos_weight, **kwargs["criterion_args"])
        self.train_auroc = AUROC()
        self.train_accuracy = Accuracy()
        self.train_precision = Precision()
        self.train_recall = Recall()
        self.val_auroc = AUROC()
        self.val_accuracy = Accuracy()
        self.val_precision = Precision()
        self.val_recall = Recall()

    def forward(self, review_batch: Tuple[Tensor, Sequence[int]], meta_batch: Tuple[Tensor, Sequence[int]]):
        return self.model(review_batch, meta_batch)

    def local_update(self, support: Sequence[Batch]):
        model = deepcopy(self.model)
        # https://stackoverflow.com/questions/63465187/runtimeerror-cudnn-rnn-backward-can-only-be-called-in-training-mode
        model.train()
        optimizer = configure_optimizers_func["AdamW"](model.parameters())
        for _ in range(self.hparams.num_inner_steps):
            for batch in support:
                y_hat, x, P = model(batch[0], batch[1])
                y = (batch[2] >= 0.75).float()
                loss = self.criterion(y_hat, y) + P
                if self.hparams.auxiliary:
                    auxiliary = self.hparams.auxiliary * self.auxiliary_loss(self.auxiliary_output_layer(x).squeeze(1), batch[3])
                    loss += auxiliary
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        return model

    def training_step(self, batch: Sequence[Tuple[Sequence[Batch], Sequence[Batch]]], *args, **kwargs) -> Tensor:
        loss_tot = 0
        y_hat_tot = []
        y_tot = []
        for support, query in batch:
            model = self.local_update(support)
            for batch in query:
                y_hat, x, P = model(batch[0], batch[1])
                y = (batch[2] >= 0.75).int()
                loss = self.criterion(y_hat, y.float()) + P
                if self.hparams.auxiliary:
                    auxiliary = self.hparams.auxiliary * self.auxiliary_loss(self.auxiliary_output_layer(x).squeeze(1), batch[3])
                    loss += auxiliary
                loss /= len(query)
                loss_tot += loss.detach()
                loss.backward()
                y_hat_tot.append(y_hat.detach())
                y_tot.append(y.detach())
            for p_global, p_local in zip(self.model.parameters(), model.parameters()):
                p_global.grad = p_local.grad if (grad := p_global.grad) is None else grad + p_local.grad
        optimizers = self.optimizers()
        optimizers.step()
        optimizers.zero_grad()
        y_hat_tot = cat(y_hat_tot)
        y_tot = cat(y_tot)
        batch_size = len(y_tot)
        # logs losses for each training_step, and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss_tot, batch_size=batch_size)
        y_hat_tot = y_hat_tot.sigmoid()
        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.train_auroc(y_hat_tot, y_tot)
        self.log("train_AUROC", self.train_auroc, batch_size=batch_size)
        self.train_accuracy(y_hat_tot, y_tot)
        self.log("train_accuracy", self.train_accuracy, batch_size=batch_size)
        self.train_precision(y_hat_tot, y_tot)
        self.log("train_precision", self.train_precision, batch_size=batch_size)
        self.train_recall(y_hat_tot, y_tot)
        self.log("train_recall", self.train_recall, batch_size=batch_size)

    def validation_step(self, batch: Sequence[Tuple[Batch, Batch]], *args, **kwargs):
        with set_grad_enabled(True):
            loss_tot = 0
            y_hat_tot = []
            y_tot = []
            for support, query in batch:
                model = self.local_update(support)
                for batch in query:
                    y_hat, x, P = model(batch[0], batch[1])
                    y = (batch[2] >= 0.75).int()
                    if self.hparams.auxiliary:
                        auxiliary = self.hparams.auxiliary * self.auxiliary_loss(
                            self.auxiliary_output_layer(x).squeeze(1), batch[3])
                    loss_tot += ((self.criterion(y_hat, y.float()) + P + auxiliary) / len(query)).detach()
                    y_hat_tot.append(y_hat.detach())
                    y_tot.append(y.detach())
            y_hat_tot = cat(y_hat_tot)
            y_tot = cat(y_tot)
            batch_size = len(y_tot)
            # logs losses at the end of the val_epoch, to logger
            self.log("val_loss", loss_tot, batch_size=batch_size)
            y_hat_tot = y_hat_tot.sigmoid()
            # logs metrics at the end of the val_epoch, to logger
            self.val_auroc(y_hat_tot, y_tot)
            self.log("val_AUROC", self.val_auroc, batch_size=batch_size)
            self.val_accuracy(y_hat_tot, y_tot)
            self.log("val_accuracy", self.val_accuracy, batch_size=batch_size)
            self.val_precision(y_hat_tot, y_tot)
            self.log("val_precision", self.val_precision, batch_size=batch_size)
            self.val_recall(y_hat_tot, y_tot)
            self.log("val_recall", self.val_recall, batch_size=batch_size)

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

@add_args_func("MeRH")
def PRH_Net_add_args(parser: ArgumentParser):
    add_args("DataModule", parser)
    add_args("PRH_Net_module", parser)
    add_args("review_representation", parser)
    parser.add_argument("-a", "--auxiliary", type=float, default=0, help="Factor of auxiliary loss.")
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument("-aa", "--auxiliary_args", type=loads, default={}, help="Auxiliary loss arguments.")
    add_args("configure_optimizers", parser)
    parser.add_argument("-as",
                        "--application_scenario",
                        type=str,
                        choices=application_scenario_module.keys(),
                        required=True,
                        help="Application scenario.")
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument("-ca", "--criterion_args", type=loads, default={}, help="Criterion args.")
    parser.add_argument("-nis", "--num_inner_steps", type=int, default=10, help="Number of inner steps.")
    fit.add_argument("-me", "--max_epochs", type=int, help="Maximum number of epochs.")

if __name__ == "__main__":
    from pytorch_lightning import Trainer

    from .utils.seed import seed

    parser = ArgumentParser(description="Product-aware helpfulness prediction of online reviews.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    fit = subparsers.add_parser("fit", help="Fit the model.")
    add_args("seed", fit)
    add_args("MeRH", fit)

    fit_from_checkpoint = subparsers.add_parser("fit_from_checkpoint", help="Fit the model from checkpoint.")
    fit_from_checkpoint.add_argument("-as",
                                     "--application_scenario",
                                     type=str,
                                     choices=application_scenario_module.keys(),
                                     required=True,
                                     help="Application scenario.")
    fit_from_checkpoint.add_argument("-p", "--path", type=str, required=True, help="Path to checkpoint.")

    save_from_checkpoint = subparsers.add_parser("save_from_checkpoint", help="Save the model from checkpoint.")
    save_from_checkpoint.add_argument("-as",
                                      "--application_scenario",
                                      type=str,
                                      choices=application_scenario_module.keys(),
                                      required=True,
                                      help="Application scenario.")
    save_from_checkpoint.add_argument("-p", "--path", type=str, required=True, help="Path to checkpoint.")

    args = parser.parse_args()

    if args.stage == "fit":
        seed(args.seed)
        model = application_scenario_module[args.application_scenario](**args.__dict__)
        # 默认num_sanity_val_steps=2，有可能全是positive导致auroc报错
        trainer = Trainer(gpus=1, default_root_dir=str(LOGS), num_sanity_val_steps=0, max_epochs=args.max_epochs)
        trainer.fit(model)
    elif args.stage == "fit_from_checkpoint":
        model = application_scenario_module[args.application_scenario].load_from_checkpoint(path := args.path)
        assert args.application_scenario == model.hparams.application_scenario
        seed(model.hparams.seed)
        # 默认num_sanity_val_steps=2，有可能全是positive导致auroc报错
        trainer = Trainer(gpus=1, default_root_dir=str(LOGS), num_sanity_val_steps=0, max_epochs=model.hparams.max_epochs)
        trainer.fit(model, ckpt_path=path)
    elif args.stage == "save_from_checkpoint":
        model = application_scenario_module[args.application_scenario].load_from_checkpoint(path := args.path)
        assert args.application_scenario == model.hparams.application_scenario
        save(model.model, f"{LOGS}/model.pth")
        if model.hparams.auxiliary:
            save(model.auxiliary_output_layer, f"{LOGS}/auxiliary_output_layer.pth")
