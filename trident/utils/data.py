from typing import Any, Optional, Union

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from trident.core import TridentDataModule
from trident.utils.hydra import instantiate_and_apply

# TODO: simple way to keep unprocessed dataset
# - should be configurable via datamodule_cfg
# - should it be possible to select by split?


def load_dataset(
    self: TridentDataModule,
    stage: Optional[str],
    config: DictConfig,
) -> None:
    # instantiate_and_apply extends `hydra.utils.instantiate` with[str, bool]
    # - _method_: call methods onto the instantiated object
    # - _apply_: call any function onto the instantiated object
    return_unprocessed = (
        OmegaConf.select(self.datamodule_cfg, "keep_raw_dataset") == True
    )
    if stage in (None, "fit"):
        self.dataset_train, self.dataset_train_raw = instantiate_and_apply(
            config.get("train", None), return_unprocessed
        )

    if stage in (None, "fit", "validate"):
        self.dataset_val, self.dataset_val_raw = instantiate_and_apply(
            config.get("val", None), return_unprocessed
        )
    if stage in (None, "test"):
        self.dataset_test, self.dataset_test_raw = instantiate_and_apply(
            config.get("test", None), self.datamodule_cfg.keep_raw_dataset
        )
    if stage in (None, "predict"):
        self.dataset_test, self.dataset_test_raw = instantiate_and_apply(
            config.get("predict", None), return_unprocessed
        )
