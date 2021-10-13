from types import MethodType
from typing import Any, List, Optional, Union

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.parsing import AttributeDict
from torch import nn
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils_base import BatchEncoding

from src.modules.mixin.evaluation import EvalMixin
from src.modules.mixin.optimizer import OptimizerMixin


class TridentModule(OptimizerMixin, EvalMixin, LightningModule):
    """Base module of Trident that wraps model, optimizer, scheduler, evaluation.

    Args:
        model (:obj:`omegaconf.dictconfig.DictConfig`):
            Needs to instantiate a :obj:`torch.nn.Module` that

            * Takes the batch unpacked
            * Returns a container with "loss" and other required attrs

            **See also:**
                
            | :py:meth:`src.modules.base.TridentModule.forward`
            | :py:meth:`src.modules.base.TridentModule.training_step`

            | **Example:** `model` of :repo:`sequence classification <configs/module/sequence_classification.yaml>`:

        optimizer (:obj:`omegaconf.dictconfig.DictConfig`):
            Configuration for the optimizer of your :py:class:`src.modules.base.TridentModule`.

            Args:
                parameters: your model's parameters

            | **Example:** :repo:`AdamW <configs/optimizer/adamw.yaml>`
            | **Implementation:** :repo:`OptimizerMixin <src/modules/mixin/optimizer.py>`

        scheduler (:obj:`omegaconf.dictconfig.DictConfig`):
            Configuration for the scheduler of the optimizer of your :py:class:`src.modules.base.TridentModule`.

            | **Example:** :repo:`Linear Warm-Up <configs/scheduler/linear_warm_up.yaml>`:
            | **Implementation:** :repo:`OptimizerMixin <src/modules/mixin/optimizer.py>`

        evaluation (:obj:`omegaconf.dictconfig.DictConfig`):
            
            Please refer to :ref:`evaluation`

            | **Example:** :repo:`(Sequence) Classification <configs/evaluation/sequence_classification.yaml>`
            | **Implementation:** :repo:`EvalMixin <src/modules/mixin/evaluation.py>`

        overrides (:obj:`omegaconf.dictconfig.DictConfig`):
            Allows you to override existing functions of the :py:class:`src.modules.base.TridentModule`.

            Example: 

            `src/my_functions/model_overrides.py`
            
            .. code-block:: python
        
                def keep_batch(self, batch: dict) -> ModelOutput:
                    return self.model(batch)

            `configs/overrides/my_model_overrides.yaml`

            .. code-block:: yaml
                
                forward:
                  _target_: src.utils.hydra.partial
                  _partial_: src.my_functions.model_overrides.keep_batch

            At last, you'd link to the overrides config as follows:

            `configs/module/my_module.yaml`

            .. code-block:: yaml
                
                defaults:
                    - /overrides: my_model_overrides


            For further details, please refer to :ref:`evaluation`

        mixins (:obj:`omegaconf.dictconfig.DictConfig`):

            **Preemptive warning: do not call super().__init__() in your mixin!**

            Mixins for the :py:class:`src.modules.base.TridentModule` essentially groups of methods.
            You can overrwrite any functionality of the module with your mixins.

            Should you require any attributes from :py:meth:`src.modules.base.TridentModule.__init__` you can access them via `self.hparams`.
            
            Examples:
                * :repo:`Evaluation <src/modules/mixin/evaluation/sequence_classification.yaml>`:

            For further details, please refer to :ref:`evaluation`


    """

    hparams: AttributeDict

    def __init__(
        self,
        model: DictConfig,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig],
        evaluation: Optional[DictConfig] = None,
        overrides: Optional[DictConfig] = None,
        mixins: Optional[DictConfig] = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.save_hyperparameters()
        # TODO(fdschmidt93): verify ordering
        LightningModule.__init__(self)
        super().__init__()
        if hasattr(self.hparams, "mixins") and self.hparams.mixins is not None:
            self.set_mixins(self.hparams.mixins)
        self.model: nn.Module = hydra.utils.instantiate(self.hparams.model)

        # for comfort:
        self.overrides = hydra.utils.instantiate(self.hparams.overrides)
        if self.overrides is not None:
            for key, value in self.overrides.items():
                setattr(self, key, MethodType(value, self))

    def set_mixins(self, mixin: list[str]) -> None:
        """Apply base class and mixins to a class instance after creation.

        Reference:
            Modified from below StackOverflow post
            Author: Ethan Furman
            URL: https://stackoverflow.com/a/8545287

        Args:
            mixin: list of imports, e.g. [src.modules.mixin.eval.EvalMixin]

        Returns:
            None:
        """
        classes: List[object] = [hydra.utils.get_class(c) for c in mixin]
        base_cls = self.__class__
        self.__class__ = type(base_cls.__name__, (*classes, base_cls), {})
        for cls_ in classes:
            cls_.__init__(self)

    def forward(self, batch: BatchEncoding) -> ModelOutput:
        """Plain forward pass of your model for which the batch is unpacked.


        **Implementation:**

            .. code-block:: python

                return self.model(**batch)
    

        Args:
            batch: input to your model

        Returns:
            ModelOutput: container with attributes required for evaluation
        """
        return self.model(**batch)

    def training_step(
        self, batch: BatchEncoding, batch_idx: int
    ) -> Union[dict[str, Any], ModelOutput]:
        """Comprises training step of your model which takes a forward pass.

        **Notes:**
            If you want to extend `training_step`, add a `on_train_batch_end` method via overrides.
            See: Pytorch-Lightning's `on_train_batch_end <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#on-train-batch-end>`

        **Implementation:**

        .. code-block:: python

            def training_step(
                self, batch: BatchEncoding, batch_idx: int
            ) -> Union[dict[str, Any], ModelOutput]:
                outputs = self(batch)
                self.log("train/loss", outputs.loss)
                return outputs

        Args:
            batch: typically comprising input_ids, attention_mask, and position_ids
            batch_idx: variable used internally by pytorch-lightning

        Returns:
            Union[dict[str, Any], ModelOutput]: model output that must have 'loss' as attr or key
        """
        outputs = self(batch)
        self.log("train/loss", outputs.loss)
        return outputs
