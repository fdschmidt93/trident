from typing import Callable, Optional

import hydra
from datasets.arrow_dataset import Dataset
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from src.utils import utils

log = utils.get_logger(__name__)


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        collator: DictConfig,
        batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.collator: Callable = hydra.utils.instantiate(collator)

    def __len__(self):
        return len(self.data_train) if self.data_train is not None else 0

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            shuffle=False,
        )
