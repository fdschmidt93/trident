from typing import List, Optional, Union, cast

import hydra
import torch
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from trident.utils.hydra import config_callbacks
from trident.utils.logging import get_logger
from trident.utils.runner import finish, log_hyperparameters

log = get_logger(__name__)


def apply_config_callbacks(cfg: DictConfig):
    if "config_callbacks" in cfg:
        log.info(
            f"Applying configuration callbacks for <{cfg.config_callbacks.keys()}>"
        )
        config_callbacks(cfg, cfg.config_callbacks)


def instantiate_objects(cfg: DictConfig, key: str) -> List[Union[Callback, Logger]]:
    objects = []
    if key in cfg and cfg[key]:
        for conf in cfg[key].values():
            if "_target_" in conf:
                log.info(f"Instantiating {key[:-1]} <{conf._target_}>")
                objects.append(hydra.utils.instantiate(conf))
    return objects


def train(cfg: DictConfig) -> Optional[torch.Tensor]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    apply_config_callbacks(cfg)

    seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    module: LightningModule = hydra.utils.instantiate(cfg.module)

    callbacks = cast(list[Callback], instantiate_objects(cfg, "callbacks"))
    logger = cast(list[Logger], instantiate_objects(cfg, "logger"))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    log_hyperparameters(cfg, module, trainer)

    trainer.fit(model=module, datamodule=datamodule)

    score = None
    if optimized_metric := cfg.get("optimized_metric"):
        score = trainer.callback_metrics.get(optimized_metric)

    if cfg.trainer.get("limit_test_batches", None) == 0:
        best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
        if isinstance(best_model_path, str):
            log.info(f"Best checkpoint path:\n{best_model_path}")
            trainer.test(module, datamodule=datamodule, ckpt_path="best")
        else:
            log.info(
                "No checkpoint callback with optimized metric in trainer. Using final checkpoint."
            )
            trainer.test(module, datamodule=datamodule)

    finish(cfg, module, datamodule, trainer, callbacks, logger)

    return score
