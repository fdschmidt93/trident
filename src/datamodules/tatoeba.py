from multiprocessing import cpu_count
from typing import Optional

from datasets.arrow_dataset import concatenate_datasets
from datasets.load import load_dataset, load_dataset_builder

from src.datamodules.base import BaseDataModule


class TatoebaDataModule(BaseDataModule):
    def __init__(
        self, lang: str, *args, **kwargs,
    ):
        # see BaseDataModule
        super().__init__(*args, **kwargs)

        self.lang: str = lang

    def prepare_data(self):
        """
        Download MNLI from Huggingface datasets hub.
        See: https://huggingface.co/datasets/glue
        """
        # download with Huggingface datasets
        dataset = load_dataset_builder("xtreme", f"tatoeba.{self.lang}")
        dataset.download_and_prepare()

    # @staticmethod
    def preprocess(self, example: dict, idx: int) -> dict:
        example["text"] = example.pop("source_sentence").strip()
        example["text_pair"] = example.pop("target_sentence").strip()
        return example

    def setup(self, stage: Optional[str] = None):
        dataset = load_dataset("xtreme", f"tatoeba.{self.lang}")
        # dataset = load_dataset("xtreme", f"tatoeba.deu")
        dataset = dataset["validation"].add_column(
            "label", range(len(dataset["validation"]))
        )
        src = dataset.remove_columns(["target_sentence", "target_lang"]).map(
            lambda x: {"lang": self.lang, "text": x["source_sentence"].strip(),}
        ).remove_columns(["source_sentence", "source_lang"])
        trg = dataset.remove_columns(["source_sentence", "source_lang"]).map(
            lambda x: {"lang": x["target_lang"], "text": x["target_sentence"].strip(),}
        ).remove_columns(["target_sentence", "target_lang"])
        if stage in (None, "test"):
            self.data_test = concatenate_datasets([src, trg])
