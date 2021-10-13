from transformers.file_utils import ModelOutput


def get_preds(outputs: ModelOutput, *args, **kwargs) -> ModelOutput:
    """Generate predictions from logits.

    Args:
        outputs: attribute dictionary comprising model output
    """
    outputs.preds = outputs.logits.argmax(dim=-1)
    return outputs
