import logging
from typing import cast

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from trident.utils.hydra import expand, harmonize_datasets
from trident.utils.logging import get_logger

log = get_logger(__name__)


def test_expand():
    """
    Tests the `expand` utility function from trident's hydra utilities.

    Key Considerations:
    1. Configuration Expansion:
        Ensures that the `expand` function correctly expands and merges configurations for specified keys.
        The main goal is to verify that specific configurations for given keys ("train", "val", and "test" in this case)
        are combined with shared configurations (like 'collate_fn', 'num_workers', and 'shuffle').

    2. Handling Special Configurations:
        It checks how the function treats configurations under special keys (like `_datasets_`). Specifically, it tests the merging of shared configurations with nested configurations under special keys.

    3. Preservation of Specific Key Configurations:
        For configurations provided under the specific keys, it ensures that these values take precedence over the shared values.
        For example, if 'shuffle' is set to 'true' under 'train' but 'false' in the shared config, the resulting 'train' config should
        have 'shuffle' set to 'true'.
    The function uses sample input and expected configurations to validate the above considerations.
    """
    # Sample input config
    input_cfg = OmegaConf.create(
        """
  collate_fn:
    _target_: val_test_collator
  num_workers: 8
  shuffle: False
  train:
    collate_fn:
      _target_: train_collator
    shuffle: true
    _datasets_:
      source:
        batch_size: 8
      target:
        batch_size: 16
  val:
    num_workers: 4
  test:
    num_workers: 2
"""
    )
    expected_cfg = OmegaConf.create(
        """
  train:
    _datasets_:
      source:
        collate_fn:
          _target_: train_collator
        num_workers: 8
        shuffle: true
        batch_size: 8
      target:
        collate_fn:
          _target_: train_collator
        num_workers: 8
        shuffle: true
        batch_size: 16
  val:
    collate_fn:
      _target_: val_test_collator
    num_workers: 4
    shuffle: false
  test:
    collate_fn:
      _target_: val_test_collator
    num_workers: 2
    shuffle: false
"""
    )
    output = expand(
        cast(DictConfig, input_cfg), ["train", "val", "test"], gen_keys=True
    )
    assert OmegaConf.structured(output) == cast(DictConfig, expected_cfg)


def test_harmonize_datasets(caplog):
    # Example configuration to test
    cfg = OmegaConf.create(
        {
            "datamodule": {
                "dataloaders": {},
                "datasets": {
                    "train": {"_datasets_": {"data1": {}, "data2": {}}},
                    "val": {},
                },
            },
            "module": {
                "evaluation": {"prepare": {}, "step_outputs": {}, "metrics": {}}
            },
        }
    )

    with caplog.at_level(logging.INFO):  # Setting the logging level to INFO
        result_cfg = harmonize_datasets(cfg)

        # Check for the specific log message
        expected_log_message = (
            "Harmonized keys: datamodule.dataloaders.train._datasets_: data1, data2"
        )
        assert expected_log_message in caplog.text

    # Assertions
    assert "train" in result_cfg.datamodule.dataloaders
    assert "data1" in result_cfg.datamodule.dataloaders.train._datasets_
    assert "data2" in result_cfg.datamodule.dataloaders.train._datasets_

    assert "train" not in result_cfg.module.evaluation.prepare
    assert "train" not in result_cfg.module.evaluation.step_outputs
    assert "train" not in result_cfg.module.evaluation.metrics

    assert "val" in result_cfg.datamodule.dataloaders
    assert "val" in result_cfg.module.evaluation.prepare
    assert "val" in result_cfg.module.evaluation.step_outputs
    assert "val" in result_cfg.module.evaluation.metrics

    assert "test" not in result_cfg.datamodule.dataloaders

    cfg = OmegaConf.create(
        {
            "datamodule": {
                "dataloaders": {},
                "datasets": {
                    "train": {},
                    "val": {"only_datasets_matters": 3, "dummy": "c"},
                    "test": {"_datasets_": {"test_a": True, "test_b": True}},
                },
            },
            "module": {
                "evaluation": {"prepare": {}, "step_outputs": {}, "metrics": {}}
            },
        }
    )

    with caplog.at_level(logging.INFO):  # Setting the logging level to INFO
        result_cfg = harmonize_datasets(cfg)

        # # Check for the specific log message
        # expected_log_message = (
        #     "Harmonized keys: datamodule.dataloaders.train._datasets_: data1, data2"
        # )
        # assert expected_log_message in caplog.text

    # Assertions
    assert "train" in result_cfg.datamodule.dataloaders
    assert "train" not in result_cfg.module.evaluation.prepare
    assert "train" not in result_cfg.module.evaluation.step_outputs
    assert "train" not in result_cfg.module.evaluation.metrics

    assert "val" in result_cfg.datamodule.dataloaders
    assert "val" in result_cfg.module.evaluation.prepare
    assert "val" in result_cfg.module.evaluation.step_outputs
    assert "val" in result_cfg.module.evaluation.metrics

    assert "test" in result_cfg.datamodule.dataloaders
    assert "test_a" in result_cfg.datamodule.dataloaders.test._datasets_
    assert "test_b" in result_cfg.datamodule.dataloaders.test._datasets_
    assert "test" in result_cfg.module.evaluation.prepare
    assert "test_a" in result_cfg.module.evaluation.prepare.test._datasets_
    assert "test_b" in result_cfg.module.evaluation.prepare.test._datasets_
