from collections import defaultdict

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


class SentenceCollator:
    def __init__(
        self, **kwargs,
    ):
        self.model_name_or_path: str = kwargs.pop("model_name_or_path")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_name_or_path
        )
        self.kwargs: dict = kwargs

    def __call__(self, inputs: dict) -> BatchEncoding:
        merged_inputs = defaultdict(list)
        for input_ in inputs:
            for k, v in input_.items():
                merged_inputs[k].append(v)
        batch: BatchEncoding = self.tokenizer(text=merged_inputs["text"], **self.kwargs)
        batch["label"] = torch.tensor(merged_inputs["label"], dtype=torch.long)
        return batch


class SentencePairCollator:
    def __init__(
        self, **kwargs,
    ):
        self.model_name_or_path: str = kwargs.pop("model_name_or_path")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_name_or_path
        )
        self.kwargs: dict = kwargs

    def __call__(self, inputs: dict) -> BatchEncoding:
        merged_inputs = defaultdict(list)
        for input_ in inputs:
            for k, v in input_.items():
                merged_inputs[k].append(v)
        batch: BatchEncoding = self.tokenizer(
            text=merged_inputs["text"],
            text_pair=merged_inputs["text_pair"],
            **self.kwargs,
        )
        batch["labels"] = torch.tensor(merged_inputs["label"], dtype=torch.long)
        return batch


# TODO update
# class PairOfSentencesCollator:
#     def __init__(
#         self, tokenizer: PreTrainedTokenizer, config: dict,
#     ):
#         self.model_name_or_path = config.pop("model_name_or_path")
#         self.config = config
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

#     def __call__(self, inputs: List[Tuple[Tuple[str, str], Tuple[str, str], int]]):
#         instance1, instance2, labels = zip(*inputs)
#         batch = {
#             "instance1": self.tokenizer(instance1),
#             "instance2": self.tokenizer(instance2),
#             "labels": torch.Tensor(labels),
#         }
#         return batch


# class PairOfSentencePairCollator:
#     def __init__(
#         self, tokenizer: PreTrainedTokenizer, max_length: int = 128,
#     ):
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def tokenize_instance(self, instance: Tuple[Tuple[str, str]]) -> dict:
#         sentence1, sentence2 = zip(*instance)
#         batch: dict = self.tokenizer(
#             text=list(sentence1),
#             text_pair=list(sentence2),
#             max_length=self.max_length,
#             padding=True,
#             return_tensors="pt",
#         )
#         return batch

#     def __call__(self, inputs: List[Tuple[Tuple[str, str], Tuple[str, str], int]]):
#         instance1, instance2, label = zip(*inputs)

#         batch = {
#             "instance1": self.tokenize_instance(instance1),
#             "instance2": self.tokenize_instance(instance2),
#             "label": torch.Tensor(label),
#         }

#         return batch
