from typing import ItemsView, Optional, Union

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import Dataset

from trident.utils.hydra import instantiate_and_apply


# TODO(fdschmidt93): add yaml example
class TridentDataset:
    """
    A container class for handling one or many datasets in Trident workflows.

    This class offers a unified approach for managing datasets created from
    trident-style Hydra configurations. It provides consistent and ergonomic access
    whether the encapsulated data comprises a single dataset or multiple datasets.

    Key Features:
    - Encapsulates one or many datasets derived from a Hydra configuration.
    - Allows access to datasets using either integer indices or string keys.
    - Internally represents a single dataset with a `None` key.
    - Abstracts the underlying data structure for seamless access.

    Parameters
    ----------
    cfg : DictConfig
        The configuration containing dataset details. If it includes multiple datasets,
        they should be nested under the "_datasets_" key.

    Attributes
    ----------
    cfg : DictConfig
        The dataset configuration. If it encapsulates multiple datasets, the top-level
        "_datasets_" key is unpacked for direct access.
    _is_datasets : bool
        Indicates if the cfg contains one or multiple datasets.
    _data : dict[Optional[str], Dataset]
        Dictionary storing datasets with their string keys. Single datasets are
        represented with a `None` key.
    _raw_data : dict[Optional[str], Dataset]
        Cache for raw datasets.
    _keys_list : list[Optional[str]]
        A list of dataset keys, offering deterministic ordering for indexing purposes.

    Methods
    -------
    get(key: Union[None, int, str]) -> Dataset:
        Retrieve a dataset using either its integer index or string key.
    get_raw(key: Union[None, int, str]) -> Dataset:
        Fetch the raw version of a dataset using either its integer index or string key.
    keys() -> list[Optional[str]]:
        Obtain all the keys representing the datasets.
    items() -> ItemsView[Optional[str], Dataset]:
        Access all (key, dataset) pairs stored in the container.
    idx2dataset() -> dict[int, Optional[str]]:
        Map integer indices to their respective dataset names.
    key_at_index(index: int) -> Optional[str]:
        Fetch the dataset name corresponding to an integer index.
    """

    def __init__(self, cfg: DictConfig):
        self._is_datasets = "_datasets_" in cfg
        self.cfg = (
            cfg if not self._is_datasets else cfg["_datasets_"]
        )  # Store config and unpack if many datasets

        # Initialize datasets for each stage using the configuration
        self._data: dict[str | None, Dataset] = {}
        self._raw_data: dict[str | None, Dataset] = {}  # Cache for raw datasets

        # Use provided helper functions to instantiate and setup datasets
        if self._is_datasets:
            for dataset_name, dataset_cfg in self.cfg.items():
                self._data[str(dataset_name)] = instantiate_and_apply(dataset_cfg)
        else:
            self._data[None] = instantiate_and_apply(self.cfg)

        # Create a list of keys for numeric indexing
        self._keys_list = list(self._data.keys())

    def _resolve_key(self, key: Union[None, int, str]) -> Union[None, str]:
        if isinstance(key, int):
            return self._keys_list[key]
        elif key is None or isinstance(key, str):
            return key
        else:
            raise TypeError(
                "`key` must be either int or one of dataset_name[`str`], `None` for many or single dataset(s), respectively!"
            )

    def get(self, key: Union[None, int, str]) -> Dataset:
        key = self._resolve_key(key)
        return self._data[key]

    def get_raw(self, key: Union[None, int, str]) -> Dataset:
        """
        Retrieve a dataset by key.

        If key is an integer, retrieve dataset by index.
        If key is a string, retrieve dataset by name.
        """
        if isinstance(key, int):
            key_name = self._keys_list[key]
            return self.get_raw(key_name)
        elif key is None or isinstance(key, str):
            if not (raw_data := self._raw_data.get(key)):
                cfg = self.cfg[key] if isinstance(key, str) else self.cfg
                raw_cfg = OmegaConf.masked_copy(
                    cfg, [str(key) for key in cfg if key not in ["_method_", "_apply_"]]
                )
                raw_data = hydra.utils.instantiate(raw_cfg)
                self._raw_data[key] = raw_data
            return raw_data
        else:
            raise TypeError("Key must be either int or str")

    def __getitem__(self, key: Union[int, str]) -> Dataset:
        """Allows indexing the wrapper directly with either int or str."""
        return self.get(key)

    def __len__(self) -> int:
        """Returns the number of unique datasets."""
        return len(self._data)

    def keys(self) -> list[Optional[str]]:
        """Return all keys (names) of the datasets."""
        return self._keys_list

    def items(self) -> ItemsView[Optional[str], Dataset]:
        """Return all items (name-dataset pairs) of the datasets."""
        return self._data.items()

    def __contains__(self, key: Union[None, str, int]) -> bool:
        if isinstance(key, str) or key is None:
            return key in self._data
        if isinstance(key, int):
            return 0 <= key < len(self._keys_list)
        return False

    def idx2dataset(self) -> dict[int, Optional[str]]:
        """
        Returns a dictionary mapping from index to dataset name.
        """
        return {i: k for i, k in enumerate(self._keys_list)}

    def key_at_index(self, index: int) -> Optional[str]:
        """Retrieve the dataset name (key) associated with the given index.

        Args:
            index (int): The index of the dataset.

        Returns:
            Optional[str]: The name (key) of the dataset at the given index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if 0 <= index < len(self._keys_list):
            return self._keys_list[index]
        raise IndexError(
            f"Index out of bounds. Valid index range: 0-{len(self._keys_list) - 1}"
        )
