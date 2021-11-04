# # Adapated from https://github.com/huggingface/transformers/tree/master/examples/pytorch

from typing import Callable

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

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

def text_classification(
    examples: dict, tokenizer: Callable, columns: dict  # {text, text_pair, labels}
) -> BatchEncoding:
    # Tokenize the texts
    text_column = columns["text"]
    text_pair_column = columns.get("text_pair", None)

    args = (
        (examples[text_column].strip(),)
        if text_pair_column is None
        else (examples[text_pair_column].strip(), examples[text_pair_column].strip())
    )
    result = tokenizer(*args)
    # Map labels to IDs (not necessary for GLUE tasks)
    # if label_to_id is not None and "label" in examples:
    #     result["label"] = [
    #         (label_to_id[l] if l != -1 else -1) for l in examples["label"]
    #     ]
    return result
