from typing import Dict, Tuple, List, Union

import numpy as np
import torch
import torchvision
import torch.nn.functional as TF
import pytorch_lightning as pl
from logging import getLogger
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

log = getLogger(__name__)


class RSNAlighningModule(pl.LightningModule):
    log_dict: Dict[str, List] = {"train": [], "val": [], "test": []}
    log_keys: Tuple[str, ...] = ("loss", "pF1")

    def __init__(self, cfg):
        super().__init__()
        self.test_results = None
        self.cfg = cfg
        self.network = self.init_model()
        self.criterion = self.init_critetrion(cfg)

    def init_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def init_criterion(self, cfg):
        loss_function = self.cfg.training.loss_function
        loss_function = loss_function.lower()
        if loss_function == 'focal_loss' or loss_function == 'fl':
            criterion = torchvision.ops.sigmoid_focal_loss()
        elif loss_function == 'binary_cross_entropy_loss' or loss_function == 'bcel':
            criterion = torch.nn.BCELoss()

        return criterion

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.cfg.training.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.training.optimizer.lr,
                weight_decay=self.cfg.training.optimizer.weight_decay
            )
        elif self.cfg.training.optimizer.name == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.training.optimizer.lr,
                weight_decay=self.cfg.training.optimizer.weight_decay
            )
        elif self.cfg.training.optimizer.name == 'Adamax':
            optimizer = torch.optim.Adamax(
                self.parameters(),
                lr=self.cfg.training.optimizer.lr,
                weight_decay=self.cfg.training.optimizer.weight_decay
            )

        else:
            raise f"{self.cfg.training.optimizer.name} optimizer is not supported."

        return optimizer

    def calc_accuracy(self, y_hat: torch.Tensor, y: torch.Tensor):
        """

        :param y_hat: (batch_size, 2, 1)
        :param y: (batch_size, 1)
        :return: score
        """
        softmax_layer = TF.softmax(y_hat, dim=1)
        y_hat = torch.max(softmax_layer, dim=1).indices
        score = ProbF1Score(y_hat, y, beta=self.cfg.training.accuracy.beta).f1_score()

        return score

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        logg = dict()
        for key in self.log_keys:
            vals = [x[key] for x in outputs if key in x.keys()]
            if len(vals) > 0:
                logg[f"train/{key}"] = torch.stack(vals).mean().item()
        self.log_dict["train"].append(logg)

    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        if isinstance(outputs[0], list):
            _outputs = []
            for out in outputs:
                _outputs += out
            outputs = _outputs

        logg = dict()
        for key in self.log_keys:
            vals = [x[key] for x in outputs if key in x.keys()]
            if len(vals) > 0:
                logg[f"val/{key}"] = torch.stack(vals).mean().item()
            self.log_dict["val"].append(logg)

        self.print_latest_metrics()

        if len(self.log_dict["val"]) > 0:
            val_loss = self.log_dict["val"][-1].get("val/loss", None)
            self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        raise NotImplementedError

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        keys = tuple(outputs[0].keys())
        results = {key: [] for key in keys}
        for d in outputs:
            for key in d.keys():
                results[key].append(d[key].cpu().numpy())

        for key in keys:
            results[key] = np.concatenate(results[key], axis=0)

        self.test_results = results

    def print_latest_metrics(self) -> None:
        # -- Logging --
        train_log = self.log_dict["train"][-1] if len(
            self.log_dict["train"]) > 0 else dict()
        val_log = self.log_dict["val"][-1] if len(
            self.log_dict["val"]) > 0 else dict()
        log_template = (
            "Epoch[{epoch:0=3}]"
            " TRAIN: loss={train_loss:>7.4f}, acc={train_acc:>7.4f}"
            " | VAL: loss={val_loss:>7.4f}, acc={val_acc:>7.4f}"
        )
        log.info(
            log_template.format(
                epoch=self.current_epoch,
                train_loss=train_log.get("train/loss", -1),
                train_acc=train_log.get("train/acc", -1),
                val_loss=val_log.get("val/loss", -1),
                val_acc=val_log.get("val/acc", -1),
            )
        )


def pf1beta(predictions, labels, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta ** 2
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


class ProbF1Score:
    def __init__(self, predictions: torch.Tensor, labels: torch.Tensor, beta: int = 1):
        self.predictions = predictions
        self.labels = labels
        self.beta = beta

        self.score = pf1beta(self.predictions, self.labels, self.beta)

    def f1_score(self):
        return self.score
