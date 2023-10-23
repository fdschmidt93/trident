from functools import cached_property
from typing import Optional, Sized, Union, cast

import hydra
from lightning import LightningDataModule, Trainer
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset, IterableDataset

from trident.core.dataset import TridentDataset
from trident.utils.enums import Split
from trident.utils.hydra import get_dataset_cfg
from trident.utils.logging import get_logger

log = get_logger(__name__)


class TridentDataModule(LightningDataModule):
    trainer: Trainer

    """
    The base class for all datamodules.

    The :obj:`TridentDataModule` facilitates writing a :obj:`LightningDataModule` with little to no boilerplate via Hydra configuration. It splits into

    - :obj:`dataset`:
    - :obj:`dataloader`:

    Args:
        dataset (:obj:`omegaconf.dictconfig.DictConfig`):

            A hierarchical :obj:`DictConfig` that instantiates or returns the dataset for :obj:`self.dataset_{train, val, test}`, respectively.

            Typical configurations follow the below pattern:

    .. seealso:: :py:func:`src.utils.hydra.instantiate_and_apply`, :py:func:`src.utils.hydra.expand`
        dataloader (:obj:`omegaconf.dictconfig.DictConfig`):


            .. seealso:: :py:func:`src.utils.hydra.expand`


    Notes:
        - The `train`, `val`, and `test` keys of :obj:`dataset` and :obj:`dataloader` join remaining configurations with priority to existing config
        - :obj:`dataloader` automatically generates `train`, `val`, and `test` keys for convenience as the config is evaluated lazily (i.e. when a :obj:`DataLoader` is requested)

    Example:

        .. code-block:: yaml

            _target_: src.datamodules.base.TridentDataModule
            _recursive_: false

            dataset:
              _target_: datasets.load.load_dataset
              # access methods of the instantiated object
              _method_:
                map: # dataset.map for e.g. tokenization
                  # kwargs for dataset.map
                  function:
                    _target_: 
                    _partial_: true
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
            dataloader:
              _target_: torch.utils.data.dataloader.DataLoader
              batch_size: 8
              num_workers: 0
              pin_memory: true
              # linked against global cfg
    """

    def __init__(
        self,
        datasets: DictConfig,
        dataloaders: DictConfig,
        cfg: Optional[DictConfig] = None,
    ):
        super().__init__()

        # variables
        self.datasets_cfg = datasets
        self.dataloaders_cfg = dataloaders
        self.datamodule_cfg = cfg

        self.datasets: dict[Split, None | TridentDataset] = {
            Split.TRAIN: None,
            Split.VAL: None,
            Split.TEST: None,
            Split.PREDICT: None,
        }

    def setup(self, stage: Optional[str] = None) -> None:
        stage_to_splits: dict[None | str, list[Split]] = {
            None: [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT],
            "fit": [Split.TRAIN, Split.VAL],
            "validate": [Split.VAL],
            "test": [Split.TEST],
            "predict": [Split.PREDICT],
        }
        for split in stage_to_splits.get(stage, []):
            if split_cfg := self.datasets_cfg.get(split.value):
                self.datasets[split] = TridentDataset(split_cfg)

    @cached_property
    def _signature_columns(self) -> Optional[list[str]]:
        import inspect

        trainer = self.trainer
        if module := getattr(trainer, "model"):
            assert hasattr(
                module, "model"
            ), f"{type(module)} does not have a `model` attribute!"
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(module.model.forward)
            _signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            _signature_columns += [
                "label",
                "label_ids",
                "labels",
                "input_ids",
                "attention_mask",
                "start_positions",
                "end_positions",
            ]
            return _signature_columns

    def __len__(self) -> int:
        """Returns the number of instances in :obj:`dataset_train`."""
        dataset_train = self.datasets[Split.TRAIN]

        if dataset_train is None:
            return 0
        elif isinstance(dataset_train, TridentDataset):
            if len(dataset_train) == 1:
                dataset = cast(Sized, dataset_train[0])
                if isinstance(dataset, IterableDataset):
                    return self.trainer.global_step
                else:
                    return len(dataset)
            else:
                return max([len(cast(Sized, d)) for _, d in dataset_train.items()])
        else:
            raise ValueError("Unexpected type for dataset_train")

    def _get_dataloader(self, split: Split) -> Union[DataLoader, CombinedLoader]:
        """Checks existence of dataset for :obj:`split` and returns :obj:`DataLoader` with cfg.

        The return type of this function typically depends on the scenario:
            * :obj:`DataLoader`: simple, single datasets
            * :obj:`CombinedLoader`: for modes, see CombinedLoader documentation
                - mode = "sequential" common in zero-shot cross-lingual transfer, evaluating on many varying datasets
                - mode = "max_size_cycle" common in zero-shot cross-lingual transfer, evaluating on many varying datasets

            .. seealso:: :py:meth:`trident.core.datamodule.TridentDataModule._get_dataloader`

        Args:
            split: one of :obj:`train`, :obj:`val`, :obj:`test`, or :obj:`predict`

        Returns:
            Union[DataLoader, list[DataLoader], dict[str, DataLoader]]: [TODO:description]
        """
        datasets = self.datasets[split]
        if datasets is None:
            raise ValueError(f"Dataset for {split.value} missing!")

        dataloaders: dict[None | str, DataLoader] = {}
        for dataset_name, dataset in datasets.items():
            if self.datamodule_cfg:
                # First, check for dataset-specific config, then fall back to top-level
                remove_unused_columns = get_dataset_cfg(
                    self.datamodule_cfg.get("remove_unused_columns"),
                    split,
                    dataset_name,
                )
                if remove_unused_columns:
                    dataset = self._remove_unused_columns(dataset, dataset_name)
            dataloader_cfg = get_dataset_cfg(self.dataloaders_cfg, split, dataset_name)
            dataloaders[dataset_name] = hydra.utils.call(
                dataloader_cfg, dataset=dataset
            )
        if None in dataloaders:
            assert len(dataloaders) == 1, "No. of dataloaders cannot exceed 1!"
            return dataloaders[None]

        if split == Split.TRAIN:
            mode = (
                self.datamodule_cfg.get(
                    "train_dataloader_combined_mode", "max_size_cycle"
                )
                if self.datamodule_cfg
                else "max_size_cycle"
            )
            return CombinedLoader(
                dataloaders,
                mode=mode,
            )

        return CombinedLoader(dataloaders, mode="sequential")

    def train_dataloader(self) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader(Split.TRAIN)

    def val_dataloader(self) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader(Split.VAL)

    def test_dataloader(self) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader(Split.TEST)

    def predict_dataloader(self) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader(Split.PREDICT)

    # TODO(fdschmidt93): maybe move out-of trident-core and into trident-xtreme
    # TODO(fdschmidt93): remove entirely? should most likely be done explicitly in configuration
    def _remove_unused_columns(
        self, dataset: Dataset, dataset_name: None | str
    ) -> Dataset:
        column_names: None | list[str] = getattr(dataset, "column_names", None)
        if column_names is not None:
            signature_columns = (
                self._signature_columns if self._signature_columns is not None else []
            )
            ignored_columns = list(set(column_names) - set(signature_columns))
            if len(ignored_columns) > 0:
                log.info(
                    f"The following columns don't have a corresponding argument in "
                    f"`{self.trainer.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."  # type: ignore
                )
            # ignoring as new_fingerprint typically not passed
            return dataset.remove_columns(ignored_columns)  # type: ignore
        else:
            dataset_name_ = dataset_name if isinstance(dataset_name, str) else "unnamed"
            log.warning(
                f"Attempting to remove unused columns for unsupported dataset {dataset_name_}!"
            )
            return dataset
