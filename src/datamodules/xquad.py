from multiprocessing import cpu_count
from typing import Optional

from datasets.load import load_dataset, load_dataset_builder

from src.datamodules.base import TridentDataModule

collator = OmegaConf.load("./configs/collator/sentence_pair.yaml")
collator['model_name_or_path'] = 'xlm-roberta-base'

class XNLIDataModule(TridentDataModule):
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
        dataset = load_dataset_builder("xtreme", f"XQuAD.{self.lang}")
        dataset.download_and_prepare()

    # @staticmethod
    # def preprocess(self, example):
    #     """
    #     Preprocess HF dataset to generically match HF tokenizers and other required preprocessing functions.

    #     Preprocess HF dataset to 
    #         1) such that input matches collators generically (`text`, `text_pair`)
    #         2) strip whitespace of examples as some instances have leading/trailing whitespaces

    #     :param example dict: example of Huggingface dataset
    #     """
        # example["text"] = example.pop("premise").strip()
        # example["text_pair"] = example.pop("hypothesis").strip()
        # return example

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            dataset = load_dataset("xtreme", f"XQuAD.{self.lang}")
            # preprocess dataset fast multithreaded
            dataset = dataset.map(self.preprocess, num_proc=cpu_count())
            self.dataset_train = dataset["train"]
            self.dataset_val = dataset["validation"]
            self.dataset_test = dataset["test"]
