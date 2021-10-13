from typing import Optional

import hydra
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule


class MetaDataModule(LightningDataModule):
    def __init__(
        self, train_dm: DictConfig, val_dm: DictConfig, test_dm: DictConfig,
    ):
        super().__init__()
        self.train_dm: LightningDataModule = hydra.utils.instantiate(train_dm)
        self.val_dm: LightningDataModule = hydra.utils.instantiate(val_dm)
        self.test_dm: LightningDataModule = hydra.utils.instantiate(test_dm)

    def setup(self, stage: Optional[str] = None):
        """Setup the train, validation, and test datamodules."""
        if stage in (None, "fit"):
            self.train_dm.setup(stage)
            self.val_dm.setup(stage)

        if stage in (None, "test"):
            self.test_dm.setup(stage)

    def train_dataloader(self):
        """Return the dataloader of the train datamodule."""
        return self.train_dm.train_dataloader()

    def val_dataloader(self):
        """Return the dataloader of the validation datamodule."""
        return self.val_dm.val_dataloader()

    def test_dataloader(self):
        """Return the dataloader of the test datamodule."""
        return self.test_dm.test_dataloader()
