import os
from typing import Optional

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.datamodules.modules.datasets import PlainDataset
from src.utils.collators import SentenceCollator


class MLDocDataModule(LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        data_dir: str = "./data/",
        train_lang: str = "english",
        train_num_samples: int = 1_000,
        dev_test_lang: str = "german",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        num_negatives: int = 3,
        samples_per_cls: int = 0,
        sample_with_replace: bool = False,
        sample_seed: int = 42,
        max_length: int = 510,
        # one of pairwise, batchwise
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_negatives = num_negatives

        self.samples_per_cls = samples_per_cls
        self.sample_with_replace = sample_with_replace
        self.sample_seed = sample_seed

        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.max_length = max_length

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_lang = train_lang
        self.train_num_samples = train_num_samples
        self.dev_test_lang = dev_test_lang

    @property
    def num_classes(self) -> int:
        return 4

    def __len__(self):
        return len(self.data_train) if self.data_train is not None else 0

    @staticmethod
    def read(
        folder: str,
        lang: str,
        split: Optional[str] = "train",
        num_samples: Optional[int] = None,
        return_dataframe: bool = False,
        samples_per_cls: Optional[int] = None,
        sample_with_replace: bool = False,
        sample_seed: int = 42,
        *args,
        **kwargs,
    ):
        if num_samples is not None:
            fname = f"{lang}.{num_samples}.{split}"
        else:
            fname = f"{lang}.{split}"
        path = os.path.join(folder, fname)
        label_groups = {
            label: i for i, label in enumerate(["CCAT", "ECAT", "GCAT", "MCAT"])
        }
        df = pd.read_csv(path, delimiter="\t", names=["label", "text"])
        df["label"] = df["label"].replace(label_groups)
        if samples_per_cls:
            df = df.groupby("label").apply(
                lambda x: x.sample(
                    min(len(x), samples_per_cls),
                    replace=sample_with_replace,
                    random_state=sample_seed,
                )
            )
        # make order consistent with other datasets
        df = df[["text", "label"]]
        return df.values.tolist() if not return_dataframe else df

    def setup(self, stage: Optional[str] = None):
        train_data = self.read(
            lang=self.train_lang,
            split="train",
            num_samples=self.train_num_samples,
            folder=self.data_dir,
            samples_per_cls=self.samples_per_cls,
            sample_with_replace=self.sample_with_replace,
            sample_seed=self.sample_seed,
        )

        self.data_train = PlainDataset(train_data)
        self.train_collator = SentenceCollator(self.tokenizer, self.max_length)

        val_data = self.read(
            lang=self.dev_test_lang,
            split="dev",
            num_samples=None,
            folder=self.data_dir,
            samples_per_cls=0,
            sample_with_replace=self.sample_with_replace,
            sample_seed=self.sample_seed,
        )
        self.data_val = PlainDataset(val_data)
        test_data = self.read(
            lang=self.dev_test_lang,
            split="test",
            num_samples=None,
            folder=self.data_dir,
            samples_per_cls=0,
            sample_with_replace=self.sample_with_replace,
            sample_seed=self.sample_seed,
        )
        self.data_test = PlainDataset(test_data)
        print(f"{len(self.data_test) =}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.train_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=SentenceCollator(self.tokenizer, self.max_length),
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=SentenceCollator(self.tokenizer, self.max_length),
            shuffle=False,
        )
