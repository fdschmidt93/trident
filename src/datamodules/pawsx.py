from multiprocessing import cpu_count
from typing import Optional

from datasets.load import load_dataset, load_dataset_builder

from src.datamodules.base import BaseDataModule

# collator = OmegaConf.load("./configs/collator/sentence_pair.yaml")
# collator['model_name_or_path'] = 'xlm-roberta-base'

class PAWSXDataModule(BaseDataModule):
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
        dataset = load_dataset_builder("xtreme", f"PAWS-X.{self.lang}")
        dataset.download_and_prepare()

    # @staticmethod
    def preprocess(self, example):
        """
        Preprocess HF dataset to generically match HF tokenizers and other required preprocessing functions.

        Preprocess HF dataset to 
            1) such that input matches collators generically (`text`, `text_pair`)
            2) strip whitespace of examples as some instances have leading/trailing whitespaces
            3) convert label to type int

        :param example dict: example of Huggingface dataset
        """
        example["text"] = example.pop("sentence1").strip()
        example["text_pair"] = example.pop("sentence2").strip()
        example["label"] = int(example["label"])
        return example

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            dataset = load_dataset("xtreme", f"PAWS-X.{self.lang}")
            # TODO possibly switch to pandas as it's faster and produces no wall of output 
            # preprocess dataset multithreaded
            dataset = dataset.map(self.preprocess, num_proc=cpu_count())
            self.data_train = dataset["train"]
            self.data_val = dataset["validation"]
            self.data_test = dataset["test"]


# class PAWSXDataModule(LightningDataModule):
#     def __init__(
#         self,
#         model_name_or_path: str,
#         lang: str = "en",
#         batch_size: int = 8,
#         data_dir: str = "./data/pawsx/",
#         num_workers: int = 8,
#         pin_memory: bool = True,
#         seed: int = 42,
#         sample_with_replace: bool = False,
#         samples_per_cls: int = 0,
#         **kwargs,
#     ):
#         super().__init__()

#         self.data_dir = Path(data_dir)
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory

#         self.samples_per_cls = samples_per_cls
#         self.sample_with_replace = sample_with_replace
#         self.seed = seed

#         self.model_name_or_path = model_name_or_path
#         self.lang = lang
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

#         self.data_train: Optional[Dataset] = None
#         self.data_val: Optional[Dataset] = None
#         self.data_test: Optional[Dataset] = None

#     def __len__(self):
#         return len(self.data_train) if self.data_train is not None else 0

#     @staticmethod
#     def read(
#         path: Union[str, Path] = "./datasets/multinli_1.0/multinli_1.0_train.jsonl",
#         samples_per_cls: int = 0,
#         sample_with_replace: bool = False,
#         seed: int = 42,
#     ):
#         df = pd.read_table(path)
#         for col in ["sentence1", "sentence2"]:
#             df[col] = df[col].str.strip()
#         if samples_per_cls:
#             df = df.groupby("label").apply(
#                 lambda x: x.sample(
#                     min(len(x), samples_per_cls),
#                     replace=sample_with_replace,
#                     random_state=seed,
#                 )
#             )
#         df = df.loc[:, ["sentence1", "sentence2", "label"]]
#         data = df.values.tolist()
#         return data

#     def setup(self, stage: Optional[str] = None):
#         target_dir = self.data_dir.joinpath(self.lang)
#         val_data = self.read(
#             path=target_dir.joinpath("dev_2k.tsv"),
#             samples_per_cls=self.samples_per_cls,
#             sample_with_replace=self.sample_with_replace,
#             seed=self.seed,
#         )
#         self.data_val = PlainDataset(val_data)
#         test_data = self.read(
#             path=target_dir.joinpath("test_2k.tsv"),
#             samples_per_cls=self.samples_per_cls,
#             sample_with_replace=self.sample_with_replace,
#             seed=self.seed,
#         )
#         self.data_test = PlainDataset(test_data)

#     def train_dataloader(self):
#         return DataLoader(
#             dataset=self.data_train,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             collate_fn=self.collator,
#             shuffle=True,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             dataset=self.data_test,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             collate_fn=self.collator,
#             shuffle=True,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             dataset=self.data_test,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             collate_fn=self.test_collator,
#             shuffle=True,
#         )
