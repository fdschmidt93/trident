from omegaconf.dictconfig import DictConfig

from src.datamodules.base import TridentDataModule
from src.utils.hydra import instantiate_and_apply


def load_dataset(self: TridentDataModule, stage: str, config: DictConfig) -> None:
    # instantiate_and_apply extends `hydra.utils.instantiate` with
    # - _method_: call methods onto the instantiated object
    # - _apply_: call any function onto the instantiated object
    if stage in (None, "fit"):
        self.dataset_train = instantiate_and_apply(config.get("train"))
    if stage in (None, "validate"):
        self.dataset_val = instantiate_and_apply(config.get("val"))
    if stage in (None, "test"):
        self.dataset_test = instantiate_and_apply(config.get("test"))
