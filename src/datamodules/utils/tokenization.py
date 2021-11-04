from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from functools import partial


class HydraTokenizer:
    def __init__(self, **kwargs):
        self.model_name_or_path = kwargs.pop("model_name_or_path")
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenize = partial(self.tokenizer, **kwargs)

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        return self.tokenize(*args, **kwargs)
