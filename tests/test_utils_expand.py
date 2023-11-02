from typing import cast

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from trident.utils.hydra import expand
from trident.utils.logging import get_logger

log = get_logger(__name__)


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


def test_expand_key_gen():
    # Sample input config
    input_cfg = OmegaConf.create(
        """
batch_size: 12
collate_fn:
  _target_: val_test_collator
num_workers: 8
shuffle: False
"""
    )
    expected_cfg = OmegaConf.create(
        """
batch_size: 12
collate_fn:
  _target_: val_test_collator
num_workers: 8
shuffle: False
train:
  batch_size: 12
  collate_fn:
    _target_: val_test_collator
  num_workers: 8
  shuffle: False
val:
  batch_size: 12
  collate_fn:
    _target_: val_test_collator
  num_workers: 8
  shuffle: False
test:
  batch_size: 12
  collate_fn:
    _target_: val_test_collator
  num_workers: 8
  shuffle: False
"""
    )
    output = expand(
        cast(DictConfig, input_cfg), ["train", "val", "test"], gen_keys=True
    )
    assert OmegaConf.structured(output) == cast(DictConfig, expected_cfg)


def test_expand_hierachy1():
    # Sample input config
    input_cfg = OmegaConf.create(
        """
batch_size: 12
collate_fn:
  _target_: val_test_collator
num_workers: 8
shuffle: False
train:
  collate_fn:
    _target_: train_collator
  shuffle: true
  _datasets_:
    first:
      batch_size: 8
      num_workers: 10
    second:
      batch_size: 16
val:
  num_workers: 4
  pin_memory: true
  _datasets_:
    source:
      shuffle: true
    target:
      num_workers: 7
test:
  num_workers: 2
  pin_memory: true
"""
    )
    expected_cfg = OmegaConf.create(
        """
batch_size: 12
collate_fn:
  _target_: val_test_collator
num_workers: 8
shuffle: False
train:
  batch_size: 12
  collate_fn:
    _target_: train_collator
  num_workers: 8
  shuffle: true
  _datasets_:
    first:
      batch_size: 8
      collate_fn:
        _target_: train_collator
      num_workers: 10
      shuffle: true
    second:
      batch_size: 16
      collate_fn:
        _target_: train_collator
      num_workers: 8
      shuffle: true
val:
  batch_size: 12
  collate_fn:
    _target_: val_test_collator
  num_workers: 4
  pin_memory: true
  shuffle: false
  _datasets_:
    source:
      batch_size: 12
      collate_fn:
        _target_: val_test_collator
      num_workers: 4
      pin_memory: true
      shuffle: true
    target:
      batch_size: 12
      collate_fn:
        _target_: val_test_collator
      num_workers: 7
      pin_memory: true
      shuffle: false
test:
  batch_size: 12
  collate_fn:
    _target_: val_test_collator
  num_workers: 2
  shuffle: false
  pin_memory: true
"""
    )
    output = expand(
        cast(DictConfig, input_cfg), ["train", "val", "test"], gen_keys=True
    )
    assert OmegaConf.structured(output) == cast(DictConfig, expected_cfg)


def test_expand_hierachy2():
    # Sample input config
    input_cfg = OmegaConf.create(
        """
batch_size: 12
collate_fn:
  _target_: val_test_collator
num_workers: 8
shuffle: False
_datasets_:
  first:
    batch_size: 8
    num_workers: 10
  second:
    batch_size: 16
train:
  collate_fn:
    _target_: train_collator
  shuffle: True
  _datasets_:
    second:
      batch_size: 6
      pin_memory: true
    third:
      pin_memory: true
val:
  num_workers: 4
  pin_memory: true
"""
    )
    expected_cfg = OmegaConf.create(
        """
batch_size: 12
collate_fn:
  _target_: val_test_collator
num_workers: 8
shuffle: False
_datasets_:
  first:
    batch_size: 8
    collate_fn:
      _target_: val_test_collator
    num_workers: 10
    shuffle: False
  second:
    batch_size: 16
    collate_fn:
      _target_: val_test_collator
    num_workers: 8
    shuffle: False
train:
  batch_size: 12 # from global cfg
  collate_fn:  # from train-level cfg
    _target_: train_collator  # from train-level cfg
  num_workers: 8 # from global-cfg
  shuffle: True  # from train-level cfg
  _datasets_:
    first:
      batch_size: 8  # from global _datasets_ cfg
      collate_fn:  # from train-level cfg
        _target_: train_collator  # from train-level cfg
      num_workers: 10  # from global _datasets_ cfg
      shuffle: True  # from train-level cfg
    second:
      batch_size: 6 # from train _datasets_ cfg
      collate_fn:  # from train-level cfg
        _target_: train_collator # from train-level cfg
      num_workers: 8 # from global cfg
      shuffle: True # from train-level cfg
      pin_memory: true # from train _datasets_ cfg
    third:
      batch_size: 12 # from global cfg
      collate_fn: # from train cfg
        _target_: train_collator # form train cfg
      num_workers: 8 # from global cfg
      shuffle: True # from train cfg
      pin_memory: true # from train _datasets_ cfg
val:
  batch_size: 12
  collate_fn:
    _target_: val_test_collator
  num_workers: 4 # val cfg
  pin_memory: true # val cfg
  shuffle: false
  _datasets_:
    first:
      batch_size: 8 # from global _datasets_
      collate_fn:
        _target_: val_test_collator
      num_workers: 4 # val cfg
      pin_memory: true # val cfg
      shuffle: false
    second:
      batch_size: 16 # from global _datasets_ cfg
      collate_fn:
        _target_: val_test_collator
      num_workers: 4 # val cfg
      pin_memory: true # val cfg
      shuffle: false
test:
  # all from global cfg
  batch_size: 12 
  collate_fn:
    _target_: val_test_collator
  num_workers: 8
  shuffle: false
  _datasets_:
    first:
      batch_size: 8
      collate_fn:
        _target_: val_test_collator
      num_workers: 10
      shuffle: False
    second:
      batch_size: 16
      collate_fn:
        _target_: val_test_collator
      num_workers: 8
      shuffle: False
"""
    )
    output = expand(
        cast(DictConfig, input_cfg), ["train", "val", "test"], gen_keys=True
    )
    assert OmegaConf.structured(output) == cast(DictConfig, expected_cfg)
