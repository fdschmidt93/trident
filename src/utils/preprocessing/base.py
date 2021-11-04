# function by task?
# how to define stuff easily upfront


# # Adapated from https://github.com/huggingface/transformers/tree/master/examples/pytorch

from dataclasses import dataclass
from functools import wraps
from typing import Callable

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


# TODO(fdschmidt93): move to proper place
# TODO(fdschmidt93): check batched true
def module_docs():
    """
    
    :code:`batched=True` returns :code:`dict[str, list]`

    The preprocessing functions are intended to be passed to `datasets.map <https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map>`_ with :code:`batched=True`.

    """

# examples = {"hypothesis": "This is my cool hypothesis", "premise": "This is another premise", "label": 1}
# s = """
# # classmethod to call
# _target_: src.utils.hydra.partial
# _partial_: transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
# # `hydra.utils.instantiate` for self in classmethod
# self:
#     _target_: transformers.AutoTokenizer.from_pretrained
#     pretrained_model_name_or_path: roberta-base
# # other kwargs for classmethod
# padding: true
# truncation: true
# max_length: 512
# """
# import hydra
# from omegaconf import OmegaConf
# def test_functions(examples, *args, **kwargs):
#     print("WORKS")
#     return examples
# t = hydra.utils.instantiate(OmegaConf.create(s))
# column_names = {"text": "hypothesis", "text_pair": "premise"}
# text_classification(examples, column_names=column_names, tokenizer=t, on_before_preprocessing=test_functions, on_after_preprocessing=test_functions)

# task_to_keys = {
#     "cola": ("sentence", None),
#     "mnli": ("premise", "hypothesis"),
#     "mrpc": ("sentence1", "sentence2"),
#     "qnli": ("question", "sentence"),
#     "qqp": ("question1", "question2"),
#     "rte": ("sentence1", "sentence2"),
#     "sst2": ("sentence", None),
#     "stsb": ("sentence1", "sentence2"),
#     "wnli": ("sentence1", "sentence2"),
# }
#     # Preprocessing the raw_datasets
#     if data_args.task_name is not None:
#         sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
#     else:
#         # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
#         non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
#         if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
#             sentence1_key, sentence2_key = "sentence1", "sentence2"
#         else:
#             if len(non_label_column_names) >= 2:
#                 sentence1_key, sentence2_key = non_label_column_names[:2]
#             else:
#                 sentence1_key, sentence2_key = non_label_column_names[0], None


def _get_columns(examples: dict, column_names: dict) -> dict:
    return {k: examples[v] for k, v in column_names.items()}


# TODO(fdschmidt93): Goal huggingface.datasets upstream PR
# TODO(fdschmidt93): add transformation cb?
# TODO(fdschmidt93): best UI? --> purely functional interface wrapped around in classes?
# TODO(fdschmidt93): do we want a partial out of this?
# stages of preprocessing
# get columns
# tokenization <- columns, tokenization args
# transform BatchEncoding <- labels
# callbacks between stages? -> OOP model -> pytorch-lightning?

# functional UI: everything required in signature

def preprocessing_hooks(func):
    """
    Hooks for preprocessing that are pre- and appended to any built-in preprocessing function.

    Args:
        on_before_preprocessing (:obj:`Callable`):
            Hook executed before the function call with the following function signature.
            
            .. code-block:: python

                def on_before_preprocessing(examples: dict, *args, **kwargs) -> dict:
                    ...
                    return examples

        on_after_preprocessing (:obj:`Callable`):
            Hook executed after the function call with the following function signature.
            
            .. code-block:: python

                def on_after_preprocessing(examples: dict, *args, **kwargs) -> BatchEncoding:
                    ...
                    return examples
    """
    @wraps(func)
    def hooks(examples: dict, *args, **kwargs):
        on_before_preprocessing = kwargs.pop("on_before_preprocessing", None)
        on_after_preprocessing = kwargs.pop("on_after_preprocessing", None)
        # avoid type checking for speed
        if on_before_preprocessing is not None:
            examples = on_before_preprocessing(examples, *args, **kwargs)
        examples = func(examples, *args, **kwargs)
        if on_after_preprocessing is not None:
            examples = on_after_preprocessing(examples, *args, **kwargs)
        return examples
    return hooks

@preprocessing_hooks
def text_classification(
    examples: dict,
    column_names: dict,  # {text, text_pair, labels}
    tokenizer: PreTrainedTokenizerBase,
) -> BatchEncoding:
    """
    Extracted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py
    """
    # Tokenize the texts
    text_column = column_names["text"]
    text_pair_column = column_names.get("text_pair", None)

    args = (
        (examples[text_column],)
        if text_pair_column is None
        else (examples[text_pair_column], examples[text_pair_column])
    )

    result = tokenizer(*args)
    # TODO(fdschmidt93): revisit
    # Map labels to IDs (not necessary for GLUE tasks)
    # if label_to_id is not None and "label" in examples:
    #     result["label"] = [
    #         (label_to_id[l] if l != -1 else -1) for l in examples["label"]
    #     ]
    return result


def token_classification(
    examples: list,
    tokenizer: PreTrainedTokenizerBase,
    padding: bool,
    label_column_name: str,
) -> BatchEncoding:
    """

    Notes:
        * -100 is `ignore_idx` for loss computation
    Credits to Huggingface's `run_ner.py <>`

    """
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length=data_args.max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(
                    label_to_id[label[word_idx]] if data_args.label_all_tokens else -100
                )
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
