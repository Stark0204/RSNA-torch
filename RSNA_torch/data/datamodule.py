from logging import getLogger

import torch
import pytorch_lightning as pl

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

logger = getLogger(__name__)


class RSNADataModule(pl.LightningDataModule):
    dataset_module: torch.utils.data.Dataset

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size

    def setup(self, stage: str):
        if stage == 'fit':
            if self.cfg.training.mode == 'train':
                self.train = self.dataset_module(self.cfg, mode='train')
                self.val = self.dataset.module(self.cfg, mode='val')
            else:
                self.train = self.dataset_module(self.cfg, mode='full')
                self.val = None
        else:
            self.train = None
            self.val = None

        if stage == 'test':
            self.test = self.dataset_module(self.cfg, mode='test')
        else:
            self.test = None

        if stage == 'predict':
            self.predict = self.dataset_module(self.cfg, mode='submission')
        else:
            self.predict = None

        logger.info(f"Dataset[train]: {self.train}")
        logger.info(f"Dataset[val]: {self.val}")
        logger.info(f"Dataset[test]: {self.test}")
        logger.info(f"Dataset[submission]: {self.predict}")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers
        )
