import logging

import torch
from hydra import compose, initialize

from trident.train import train
from trident.utils.logging import get_logger

log = get_logger(__name__)

CONFIG_PATH = "./configs/"


def test_single_train_dataloader() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(config_name="test_single_train_single_val_test")
        is_ = torch.Tensor(train(cfg))
        should = torch.zeros(1)[0]
        assert torch.allclose(should, is_), f"{is_.item()} is not {should}"


def test_multi_train_dataloader(caplog) -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(config_name="test_many_train_single_val_test")
        with caplog.at_level(logging.WARNING):
            out = train(cfg)
            assert (
                "Attempting to remove unused columns for unsupported dataset first_half!"
                in caplog.text
            )
            assert (
                "Attempting to remove unused columns for unsupported dataset second_half!"
                in caplog.text
            )
        is_ = torch.Tensor(out)
        should = torch.zeros(1)[0]
        assert torch.allclose(should, is_), f"{is_.item()} is not {should}"


def test_off_by_one_test_single_train_many_val_test_single_eval_config() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(config_name="test_single_train_many_val_test_single_eval_config")
        is_ = torch.Tensor(train(cfg))
        should = torch.Tensor([1])
        assert torch.allclose(should, is_), f"{is_.item()} is not {should}"


def test_off_by_two_test_single_train_many_val_test_single_eval_config() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(
            config_name="test_single_train_many_val_test_single_eval_config",
            overrides=["optimized_metric='off_by_two/val/mse_loss'"],
        )
        is_ = torch.Tensor(train(cfg))
        should = torch.Tensor([4])
        assert torch.allclose(should, is_), f"{is_.item()} is not {should}"


def test_off_by_one_test_single_train_many_val_test_single_eval_configs() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(config_name="test_single_train_many_val_test_many_eval_configs")
        is_ = torch.Tensor(train(cfg))
        should = torch.Tensor([1])
        assert torch.allclose(should, is_), f"{is_.item()} is not {should}"


def test_off_by_two_test_single_train_many_val_test_many_eval_config() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(
            config_name="test_single_train_many_val_test_many_eval_configs",
            overrides=["optimized_metric='off_by_two/val/mse_loss'"],
        )
        is_ = torch.Tensor(train(cfg))
        should = torch.Tensor([4])
        assert torch.allclose(should, is_), f"{is_.item()} is not {should}"
