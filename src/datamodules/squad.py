from functools import partial
from multiprocessing import cpu_count
from typing import Optional

from datasets.load import load_dataset, load_dataset_builder
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.datamodules.base import TridentDataModule
from src.utils import flatten_dict
from src.utils.qa import prepare_train_features

# collator = OmegaConf.load("./configs/collator/sentence_pair.yaml")
# collator['model_name_or_path'] = 'xlm-roberta-base'

# TypeError: squad_convert_examples_to_features() missing 5 required positional arguments: 'tokenizer', 'max_seq_length', 'doc_stride', 'max_query_length', and 'is_training'

# tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# fn = partial(prepare_train_features, tokenizer=tokenizer)

# def batch(iterable, n=1000):
#     l = len(iterable)
#     for ndx in range(0, l, n):
#         yield iterable[ndx:min(ndx + n, l)]

# for i, b in enumerate(batch(dataset['train'])):
#     x = fn(b)
#     if len(x['attention_mask']) != 1000:
#         break

# tokenized_examples = tokenizer(
#     b['question'],
#     b['context'],
#     truncation="only_second",
#     max_length=512,
#     stride=128,
#     return_overflowing_tokens=False,
#     return_offsets_mapping=True,
#     padding="max_length",
# )

# kwargs = {
#     "examples": examples,
#     "tokenizer": AutoTokenizer.from_pretrained("prajjwal1/bert-tiny"),
#     "is_training": True,
#     "max_seq_length": 128,
#     "doc_stride": 1,
#     "max_query_length": 100,
# }

# examples = [SquadExample(
#             qas_id = example.get("id"),
#             title = example.get("title"),
#             question_text = example.get("question"),
#             context_text = example.get("context"),
#             answer_text = example.get("answer_text"),
#             is_impossible = example.get("is_impossible"),
#             start_position_character = example.get("start_position_character")
#     ) for example in b_['train']]

# squad_convert_examples_to_features(**kwargs)


class SQuADDataModule(TridentDataModule):
    def __init__(
        self, *args, **kwargs,
    ):
        # see BaseDataModule
        super().__init__(*args, **kwargs)


    def prepare_data(self):
        """
        Download MNLI from Huggingface datasets hub.
        See: https://huggingface.co/datasets/glue
        """
        # download with Huggingface datasets
        dataset = load_dataset_builder("squad")
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

    # @staticmethod
    # def process(self, example):
    #     return prepare_train_features(example, self.tokenizer, self.column_names)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            dataset = load_dataset("squad")
            self.dataset_train = dataset["train"]
            self.dataset_val = dataset["validation"]
