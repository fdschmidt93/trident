import torch
from hydra import compose, initialize

from trident.train import train

CONFIG_PATH = "./configs/"


def test_single_train_dataloader() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(config_name="test_single_train_single_val_test")
        assert torch.allclose(torch.zeros(1)[0], torch.Tensor(train(cfg)))


def test_multi_train_dataloader() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(config_name="test_many_train_single_val_test")
        assert torch.allclose(torch.zeros(1)[0], torch.Tensor(train(cfg)))


def test_off_by_one_test_single_train_many_val_test_single_eval_config() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(config_name="test_single_train_many_val_test_single_eval_config")
        assert torch.allclose(torch.Tensor([1]), torch.Tensor(train(cfg)))


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
        assert torch.allclose(torch.Tensor([4]), torch.Tensor(train(cfg)))


def test_off_by_one_test_single_train_many_val_test_single_eval_configs() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(config_name="test_single_train_many_val_test_many_eval_configs")
        assert torch.allclose(torch.Tensor([1]), torch.Tensor(train(cfg)))


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
        assert torch.allclose(torch.Tensor([4]), torch.Tensor(train(cfg)))
