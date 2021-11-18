import sys
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer, seed_everything)
from pytorch_lightning.loggers import LightningLoggerBase
from trident.utils.hydra import config_callback
from trident.utils.log import get_logger

from src.utils.runner import finish, log_hyperparameters

log = get_logger(__name__)


def train(cfg: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Init lightning datamodule

    if "config_callback" in cfg:
        log.info(f"Applying configuration callbacks for <{cfg.config_callback.keys()}>")
        config_callback(cfg, cfg.config_callback)

    if "imports" in cfg:
        if isinstance(cfg.imports, str):
            cfg.imports = [cfg.imports]
        for path in cfg.imports:
            sys.path.insert(0, path)
            log.info(f"{path} added to PYTHONPATH")

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage=None)

    # Init lightning model
    log.info(f"Instantiating model <{cfg.module._target_}>")
    module: LightningModule = hydra.utils.instantiate(cfg.module)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    log_hyperparameters(
        cfg=cfg,
        module=module,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if cfg.get("train", True):
        log.info("Starting training!")
        trainer.fit(model=module, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if cfg.get("test_after_training") and not cfg.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        if cfg.get("train", True):
            trainer.test()
        else:
            trainer.test(module, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    finish(
        cfg=cfg,
        module=module,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
