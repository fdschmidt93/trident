from typing import Optional

import hydra
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule


class MetaDataModule(LightningDataModule):
    """DataModule of DataModules to train, validate, and test on varying datasets.

    Args:
        train_dm, val_dm, test_dm (:obj:`omegaconf.dictconfig.DictConfig`):
        
            Configurations for datamodules that are passed accordingly to `{train, val, test}_dataloader`.

            | **Example:** :repo:`pawsx <configs/experiment/pawsx.yaml>`
    """
    def __init__(
        self, train_dm: DictConfig, val_dm: DictConfig, test_dm: DictConfig,
    ):
        super().__init__()
        self.train_dm: LightningDataModule = hydra.utils.instantiate(train_dm)
        self.val_dm: LightningDataModule = hydra.utils.instantiate(val_dm)
        self.test_dm: LightningDataModule = hydra.utils.instantiate(test_dm)

    def setup(self, stage: Optional[str] = None):
        """Setup the train, validation, and test datamodules with their original setup functions."""
        if stage in (None, "fit"):
            self.train_dm.setup(stage)
            self.val_dm.setup(stage)

        if stage in (None, "test"):
            self.test_dm.setup(stage)

    def train_dataloader(self):
        """Returns the dataloader of the train datamodule."""
        return self.train_dm.train_dataloader()

    def val_dataloader(self):
        """Returns the dataloader of the validation datamodule."""
        return self.val_dm.val_dataloader()

    def test_dataloader(self):
        """Returns the dataloader of the test datamodule."""
        return self.test_dm.test_dataloader()
