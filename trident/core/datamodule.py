from types import MethodType
from typing import Optional

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from trident.utils.logging import get_logger

log = get_logger(__name__)

# TODO(fdschmidt93): link dataloader against global cfg
# TODO(fdschmidt93): add docs for {train, val, test} options
# TODO(fdschmidt93): handle predict in more principled fashion
# TODO(fdschmidt93): update documentation once preprocessing routines are set
# TODO(fdschmidt93): document partial trident functions


class TridentDataModule(LightningDataModule):
    """
    The base class for all datamodules.

    The :obj:`TridentDataModule` facilitates writing a :obj:`LightningDataModule` with little to no boilerplate via Hydra configuration. It splits into

    - :obj:`dataset_cfg`:
    - :obj:`dataloader_cfg`:

    Args:
        dataset_cfg (:obj:`omegaconf.dictconfig.DictConfig`):

            A hierarchical :obj:`DictConfig` that instantiates or returns the dataset for :obj:`self.dataset_{train, val, test}`, respectively.

            Typical configurations follow the below pattern:

    .. seealso:: :py:func:`src.utils.hydra.instantiate_and_apply`, :py:func:`src.utils.hydra.expand`
        dataloader_cfg (:obj:`omegaconf.dictconfig.DictConfig`):


            .. seealso:: :py:func:`src.utils.hydra.expand`

        seed (:obj:`int`, `optional`):
            Linked against `config.seed` by default for convenience and maybe used
            for functionality that is not yet set by :obj:`pytorch_lightning.seed_everything`,
            which sets the seed for `pytorch`, `numpy` and `python.random`.

    Notes:
        - The `train`, `val`, and `test` keys of :obj:`dataset_cfg` and :obj:`dataloader_cfg` join remaining configurations with priority to existing config
        - :obj:`dataloader_cfg` automatically generates `train`, `val`, and `test` keys for convenience as the config is evaluated lazily (i.e. when a :obj:`DataLoader` is requested)

    Example:

        .. code-block:: yaml

            _target_: src.datamodules.base.TridentDataModule
            _recursive_: false

            dataset_cfg:
              _target_: datasets.load.load_dataset
              # access methods of the instantiated object
              _method_:
                map: # dataset.map for e.g. tokenization
                  # kwargs for dataset.map
                  function:
                    _target_: src.utils.hydra.partial
                    _partial_: src.datamodules.preprocessing.{...}
                  num_proc: 12

              path: glue
              name: mnli
              train:
                split: "train"
              val:
                # inherits `path`, `name`, etc.
                split: "validation_mismatched+validation_matched"
              test:
                # set `path`, `name`, `lang` specifically, remainder inherited
                path: xtreme
                name: xnli
                lang: de
                split: "test"

            dataloader_cfg:
              _target_: torch.utils.data.dataloader.DataLoader
              batch_size: 8
              num_workers: 0
              pin_memory: true
              # linked against global cfg
              seed: ${seed}
    """

    def __init__(
        self,
        datamodule_cfg: DictConfig,
        dataset_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        seed: int = 42,
        overrides: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.seed = seed

        # variables
        self.datamodule_cfg = datamodule_cfg
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg

        self.dataset_train: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_test: Optional[Dataset] = None
        self.dataset_predict: Optional[Dataset] = None

        self.dataset_train_raw: Optional[Dataset] = None
        self.dataset_val_raw: Optional[Dataset] = None
        self.dataset_test_raw: Optional[Dataset] = None
        self.dataset_predict_raw: Optional[Dataset] = None

        # # methods via hydra
        # for method in [
        #     "transfer_batch_to_device",
        #     "on_before_batch_transfer",
        #     "on_after_batch_transfer",
        # ]:
        #     if (
        #         m := hydra.utils.instantiate(self.datamodule_cfg.get(method, None))
        #         is not None
        #     ):
        #         setattr(self.datamodule_cfg, method, m)

        self.overrides = hydra.utils.instantiate(overrides)
        if self.overrides is not None:
            for key, value in self.overrides.items():
                setattr(self, key, MethodType(value, self))

    def __len__(self) -> int:
        """Returns the number of instances in :obj:`dataset_train`."""
        return len(self.dataset_train) if self.dataset_train is not None else 0

    def prepare_data(self) -> None:
        """
        .. seealso:: `LightningDataModule.prepare_data <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#prepare-data>`_
        """
        hydra.utils.call(self.datamodule_cfg.get("prepare_data", None), self)

    # TODO(fdschmidt93): enable overrides
    # def transfer_batch_to_device(self, batch, dataloader_idx: int) -> None:
    #     """
    #     .. seealso:: `LightningDataModule.transfer_batch_to_device <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#transfer-batch-to-device>`_
    #     """
    #     self.datamodule_cfg.transfer_batch_to_device(self, batch, dataloader_idx)

    # def on_before_batch_transfer(self, batch, dataloader_idx: int) -> None:
    #     """
    #     .. seealso:: `LightningDataModule.on_before_batch_transfer <https://pytorch_lightning.readthedocs.io/en/latest/extensions/datamodules.html#on-before-batch-transfer>`_
    #     """
    #     self.datamodule_cfg.on_before_batch_transfer(self, batch, dataloader_idx)

    # def on_after_batch_transfer(self, batch, dataloader_idx: int) -> None:
    #     """
    #     .. seealso:: `LightningDataModule.on_after_batch_transfer <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#on-after-batch-transfer>`_
    #     """
    #     self.datamodule_cfg.on_after_batch_transfer(self, batch, dataloader_idx)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up `self.dataset_{train, val, test}` datasets that are fed to the corresponding :obj:`DataLoader` instances.

        :obj:`dataset_cfg` must point to a function with the following specification. The below example shows a naive use case:

        .. code-block:: python

            def setup(cfg: DictConfig, datamodule: TridentDataModule, stage: Optional[str]) -> None:
                # load and process the dataset as defined in dataset_cfg.my_dataset configuration
                dataset = hydra.utils.instantiate(cfg.my_dataset)
                datamodule.dataset_train = dataset["train"]
                datamodule.dataset_val = dataset["val"]
                datamodule.dataset_test = dataset["test"]

        The corresponding :obj:`datamodule` configuration may then look as follows:


        .. code-block:: yaml

            dataset_cfg:
                _target_: src.utils.hydra.partial
                _partial_: src.custom.my_setup_func

                my_dataset:
                    _target_: src.custom.dataset.MyDataset
                    ...

        Args:
            self: datamodule
            stage: one of "fit", "validate", "test"

        .. seealso:: `LightningDataModule.setup <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup>`_, :py:func:`src.datamodules.utils.load_dataset`, :py:func:`src.utils.hydra.instantiate_and_apply`
        """
        hydra.utils.call(
            self.datamodule_cfg.setup, self, stage, getattr(self, "dataset_cfg", None)
        )
        self.on_after_setup()

    # TODO
    def on_before_setup(self):
        hydra.utils.call(getattr(self.datamodule_cfg, "on_before_setup", None), self)

    def on_after_setup(self):
        hydra.utils.call(getattr(self.datamodule_cfg, "on_after_setup", None), datamodule=self)

    def _get_dataloader(self, split: str) -> DataLoader:
        """Checks existence of dataset for :obj:`split` and returns :obj:`DataLoader` with cfg."""
        dataset = getattr(self, f"dataset_{split}")
        assert dataset is not None, f"Dataset for {split} missing!"
        if OmegaConf.select(self.datamodule_cfg, "remove_unused_columns"):
            dataset = self._remove_unused_columns(dataset)
        return hydra.utils.call(getattr(self, "dataloader_cfg")[split], dataset=dataset)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader("predict")

    # TODO(fdschmidt93): maybe move out-of trident-core and into trident-xtreme
    def _remove_unused_columns(
        self,
        dataset: Dataset,
    ):
        import inspect

        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(self.trainer.model.model.forward)
        self._signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        self._signature_columns += ["label", "label_ids"]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            log.info(
                f"The following columns don't have a corresponding argument in "
                f"`{self.trainer.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )
        return dataset.remove_columns(ignored_columns)