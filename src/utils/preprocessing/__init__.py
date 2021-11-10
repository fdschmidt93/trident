# function by task?
# how to define stuff easily upfront


# # Adapated from https://github.com/huggingface/transformers/tree/master/examples/pytorch

from dataclasses import dataclass
from functools import partial, wraps
from typing import Callable, Union

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


# # TODO(fdschmidt93): move to proper place
# # TODO(fdschmidt93): check batched true
# def module_docs():
#     """
#     :code:`batched=True` returns :code:`dict[str, list]`
#     The preprocessing functions are intended to be passed to `datasets.map <https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map>`_ with :code:`batched=True`.
#     """

# examples = {"hypothesis": "This is my cool hypothesis", "premise": "This is another premise", "label": 1}

s = """
# classmethod to call
_target_: src.utils.hydra.partial
_partial_: transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
# `hydra.utils.instantiate` for self in classmethod
self:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: roberta-base
# other kwargs for classmethod
padding: true
truncation: true
max_length: 512
"""
import hydra
from omegaconf import OmegaConf


def test_functions(examples, *args, **kwargs):
    print("WORKS")
    return examples
t = hydra.utils.instantiate(OmegaConf.create(s))
column_names = {"text": "hypothesis", "text_pair": "premise"}

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
    column_names: dict,
    label_all_tokens: bool = True,
    # TODO(fdschmidt93): factor out
    is_split_into_words: bool = True,
) -> BatchEncoding:
    """

    Notes:
        * -100 is `ignore_idx` for loss computation
    Credits to Huggingface's `run_ner.py <>`

    """

    tokenized_inputs = tokenizer(
        examples[text_column_name],
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
                    label_to_id[label[word_idx]] if label_all_tokens else -100
                )
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs



def question_answering_train(
    examples,
    tokenizer: Union[partial, PreTrainedTokenizerBase],
    column_names: dict,
) -> BatchEncoding:

    question_column_name = column_names.get("question")
    context_column_name = column_names.get("context")
    answer_column_name = column_names.get("answers")
    
    # workaround to comply with trident-hydra configuration seemleassly
    if isinstance(tokenizer, partial) and isinstance(tokenizer.args[0], PreTrainedTokenizerBase):
        pad_on_right = tokenizer.args[0].padding_side == "right"
    elif isinstance(tokenizer, PreTrainedTokenizerBase):
        pad_on_right = t.padding_side == "right"
    else:
        pad_on_right = True

    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [
        q.lstrip() for q in examples[question_column_name]
    ]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # import pudb
    # pu.db
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.args[0].cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def question_answering_val_test(
    examples, tokenizer: PreTrainedTokenizerBase,
    column_names: dict,
) -> BatchEncoding:
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    question_column_name = column_names.get("question")
    context_column_name = column_names.get("context")

    examples[question_column_name] = [
        q.lstrip() for q in examples[question_column_name]
    ]

    # workaround to comply with trident-hydra configuration seemleassly
    if isinstance(tokenizer, partial) and isinstance(tokenizer.args[0], PreTrainedTokenizerBase):
        pad_on_right = tokenizer.args[0].padding_side == "right"
    elif isinstance(tokenizer, PreTrainedTokenizerBase):
        pad_on_right = t.padding_side == "right"
    else:
        pad_on_right = True

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        text=examples[question_column_name if pad_on_right else context_column_name],
        text_pair=examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def multiple_choice(examples):
    first_sentences = [[context] * 4 for context in examples[context_name]]
    question_headers = examples[question_header_name]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    # Flatten out
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    # Un-flatten
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

# steps
# 1. cleanup preprocessing
# 2. port datamodule cfgs
# 3. evaluation

import hydra
from omegaconf import OmegaConf
from datasets.load import load_dataset
from src.utils.hydra import instantiate_and_apply
s = """
_target_: datasets.load.load_dataset
_method_:
    map:
        function:
            _target_: src.utils.hydra.partial
            _partial_: src.utils.preprocessing.question_answering_train
            column_names:
                question: question
                context: context
                answers: answers
            tokenizer:
                _target_: src.utils.hydra.partial
                _partial_: transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
                self:
                    _target_: transformers.AutoTokenizer.from_pretrained
                    pretrained_model_name_or_path: "roberta-base"
                max_length: 512
                stride: 128
                return_overflowing_tokens: true
                return_offsets_mapping: true
        remove_columns:
            - "id"
            - "title"
            - "context"
            - "question"
            - "answers"
        batched: true
path: "squad"
split: "train"
"""
cfg = OmegaConf.create(s)
ds = instantiate_and_apply(cfg)
# ds = load_dataset("squad", split="train")
# # columns = ds.column_names
# # x = ds.map(preprocess, remove_columns=columns, batched=True)
# x = ds.map(preprocess, batched=True)
