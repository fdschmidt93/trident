from functools import partial
from typing import Callable, Optional, Union

from datasets.arrow_dataset import Dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from datasets.load import load_dataset, load_dataset_builder

import src.datamodules.utils.preprocessing as preprocessing
from src.datamodules.base import TridentDataModule


class MNLIDataModule(TridentDataModule):
    """
    :Name: Multi-Genre Natural Language Inference (MultiNLI)
    :Task: Text Classification
    :Homepage: https://cims.nyu.edu/~sbowman/multinli/
    :Paper: https://cims.nyu.edu/~sbowman/multinli/paper.pdf
    :datasets: https://huggingface.co/datasets/glue#mnli
    """
    def __init__(
        self, *args, **kwargs,
    ):
        # see BaseDataModule
        super().__init__(*args, **kwargs)

    @staticmethod
    def prepare_data():
        """
        Download MNLI from Huggingface datasets hub.
        See: https://huggingface.co/datasets/glue
        """
        # download with Huggingface datasets
        dataset = load_dataset_builder("glue", "mnli")
        dataset.download_and_prepare()

    @staticmethod
    def preprocess(example):
        """
        Preprocess HF dataset to generically match HF tokenizers and other required preprocessing functions.

        Preprocess HF dataset to 
            1) such that input matches collators generically (`text`, `text_pair`)
            2) strip whitespace of examples as some instances have leading/trailing whitespaces

        :param example dict: example of Huggingface dataset
        """
        example["text"] = example.pop("premise").strip()
        example["text_pair"] = example.pop("hypothesis").strip()
        return example

    def setup(self, stage: Optional[str] = None):
        """Sets up the MNLI dataset."""
        self.dataset = self.load_dataset(split = "train")

        if stage in (None, "fit"):
            self.dataset_train = self.dataset["train"]

        if stage in (None, "test"):
            dataset = load_da
            self.dataset_val = concatenate_datasets(
                [dataset["validation_mismatched"], dataset["validation_matched"]]
            )
            # if stage in (None, "test"):
            self.dataset_test = self.dataset_val
