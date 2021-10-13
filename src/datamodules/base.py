from abc import ABC, abstractmethod
from types import MethodType
from typing import Callable, Optional

import hydra
from datasets.arrow_dataset import Dataset
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader


class BaseDataModule(LightningDataModule, ABC):
    """
    The base class for all datamodules.

    TODO:
        * Support Sampler

    Args:
        collate_fn (:obj:`omegaconf.dictconfig.DictConfig`):
            Needs to return a :obj:`Callable` that processes a batch returned by the :obj:`DataLoader`.

            .. seealso:: :py:meth:`src.modules.base.TridentModule.forward`, :py:meth:`src.modules.base.TridentModule.training_step`, :repo:`MNLI config <configs/datamodule/mnli.yaml>`

        batch_size (:obj:`int`):
            The batch size returned by your :obj:`DataLoader`
            
            .. seealso:: `DataLoader documentation <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_

        num_workers (:obj:`int`):
            The number of workers for your :obj:`DataLoader`

            .. seealso:: `DataLoader documentation <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_

        train_collate_fn (:obj:`omegaconf.dictconfig.DictConfig`, `optional`):
            If passed, replaces `collate_fn` for the train dataloader.

        val_collate_fn (:obj:`omegaconf.dictconfig.DictConfig`, `optional`):
            If passed, replaces `collate_fn` for the val dataloader.

        test_collate_fn (:obj:`omegaconf.dictconfig.DictConfig`, `optional`):
            If passed, replaces `collate_fn` for the test dataloader.

        seed (:obj:`int`, `optional`):
            Linked against `config.seed` by default for convenience and maybe used
            for functionality that is not yet set by :obj:`pytorch_lightning.seed_everything`,
            which sets the seed for `pytorch`, `numpy` and `python.random`.

    """

    def __init__(
        self,
        collate_fn: DictConfig,
        batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        overrides: Optional[DictConfig] = None,
        train_collate_fn: Optional[DictConfig] = None,
        val_collate_fn: Optional[DictConfig] = None,
        test_collate_fn: Optional[DictConfig] = None,
        seed: int = 42,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self.dataset: Optional[Dataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.collate_fn: Callable = hydra.utils.instantiate(collate_fn)
        self.train_collate_fn: Callable = hydra.utils.instantiate(train_collate_fn)
        self.val_collate_fn: Callable = hydra.utils.instantiate(val_collate_fn)
        self.test_collate_fn: Callable = hydra.utils.instantiate(test_collate_fn)

        self.overrides = hydra.utils.instantiate(overrides)
        if self.overrides is not None:
            for key, value in self.overrides.items():
                setattr(self, key, MethodType(value, self))

    def __len__(self):
        return len(self.data_train) if self.data_train is not None else 0

    # TODO(fdschmidt93): revisit
    # @abstractmethod
    def setup(self):
        """Sets up `self.data_{train, val, test}` datasets that are fed to the corresponding :obj:`DataLoader` instances.

        Typically wraps `datasets <https://huggingface.co/docs/datasets/>`_ in `setup` method of dataset.

        .. code-block:: python

            def setup(self, stage: Optional[str] = None):
                    if stage in (None, "fit"):
                        dataset = load_dataset("glue", "mnli")
                        dataset = dataset.map(self.preprocess, num_proc=cpu_count())
                        dataset = dataset.rename_column("label", "labels")
                        self.data_train = dataset["train"]
                        self.data_val = concatenate_datasets(
                            [dataset["validation_mismatched"], dataset["validation_matched"]]
                        )
                        # if stage in (None, "test"):
                        self.data_test = self.data_val

        Args:
            self: datamodule

        Raises:
            NotImplementedError: if method is not implemented in sub-classes

        .. seealso:: :py:meth:`src.datamodules.mnli.MNLIDataModule.setup`
        """
        raise NotImplementedError(f"Please implement setup for {type(self)}")

    # TODO(fdschmidt93): support custom sampler
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.train_collate_fn
            if self.train_collate_fn is not None
            else self.collate_fn,
            shuffle=True,
        )

    # TODO(fdschmidt93): support custom sampler
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.val_collate_fn
            if self.val_collate_fn is not None
            else self.collate_fn,
            shuffle=False,
        )

    # TODO(fdschmidt93): support custom sampler
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.test_collate_fn
            if self.test_collate_fn is not None
            else self.collate_fn,
            shuffle=False,
        )
