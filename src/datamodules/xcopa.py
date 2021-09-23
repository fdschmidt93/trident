from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.datamodules.modules.datasets import PlainDataset


class XCOPADataModule(LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 8,
        data_dir: str = "./data/xcopa/",
        train_lang: str = "en",
        val_test_lang: str = "et",
        max_length: int = 100,
        num_negatives: int = 3,
        num_workers: int = 8,
        pin_memory: bool = True,
        seed: int = 42,
        sample_with_replace: bool = False,
        samples_per_cls: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_negatives = num_negatives

        self.samples_per_cls = samples_per_cls
        self.sample_with_replace = sample_with_replace
        self.seed = seed

        self.model_name_or_path = model_name_or_path
        self.train_lang = train_lang
        self.val_test_lang = val_test_lang
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 3

    def __len__(self):
        return len(self.data_train) if self.data_train is not None else 0

    @staticmethod
    def read(
        path: Union[str, Path] = "./datasets/multinli_1.0/multinli_1.0_train.jsonl",
        samples_per_cls: int = 0,
        sample_with_replace: bool = False,
        seed: int = 42,
    ):
        df = pd.read_json(path, lines=True)
        for col in ["premise", "choice1", "choice2"]:
            df[col] = df[col].str.strip()
        if samples_per_cls:
            df = df.groupby("label").apply(
                lambda x: x.sample(
                    min(len(x), samples_per_cls),
                    replace=sample_with_replace,
                    random_state=seed,
                )
            )
        df = df.loc[:, ["premise", "choice1", "choice2", "label"]]
        data = df.values.tolist()
        return data

    def setup(self, stage: Optional[str] = None):
        target_dir = self.data_dir.joinpath(self.lang)
        train_data = self.read(
            path=target_dir.joinpath(f"train.{self.train_lang}.jsonl"),
            samples_per_cls=self.samples_per_cls,
            sample_with_replace=self.sample_with_replace,
            seed=self.seed,
        )
        self.data_train = PlainDataset(train_data)
        val_data = self.read(
            path=target_dir.joinpath(f"val.{self.dev_test_lang}.jsonl"),
            samples_per_cls=self.samples_per_cls,
            sample_with_replace=self.sample_with_replace,
            seed=self.seed,
        )
        self.data_val = PlainDataset(val_data)
        test_data = self.read(
            path=target_dir.joinpath("test_2k.tsv"),
            path=target_dir.joinpath(f"test.{self.dev_test_lang}.jsonl"),
            samples_per_cls=self.samples_per_cls,
            sample_with_replace=self.sample_with_replace,
            seed=self.seed,
        )
        self.data_test = PlainDataset(test_data)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.test_collator,
            shuffle=True,
        )
