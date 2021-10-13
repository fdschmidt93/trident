.. _customization:

Customization
=============

.. _add-path:

Adding your project to path
---------------------------

You can add entire projects or single python files to the environment of trident by providing your paths as a yaml file in `/config/imports`, as below:

:obj:`/configs/imports/my_project.yaml`

.. code-block:: yaml

    # Union[str, list[str]]
    - /path/to/your/python/module

This allows you to later on natively instantiate your models as if they were part of the project.

.. _link-function:

Linking your own function
-------------------------

Most often, you would link your own functions to customize the evaluation loop. The below example is a common pattern in customized evaluation.

.. code-block:: yaml

    # see also /configs/evaluation/sequence_classification.yaml
    apply:
      batch: null
      outputs: 
        # required _target_ for hydra.utils.instantiate
        _target_: src.utils.hydra.partial
        # actual function
        _partial_: src.evaluation.classification.get_preds
      step_outputs: null

:obj:`partial` is a wrapper around :obj:`functools.partial`.

.. _function-override:

Function Overrides
------------------

You can override functions of the model and datamodule explicitly as follows.

1. Write your custom function the project or python path (see :ref:`add-path`)
2. Provide a yaml configuration in `/configs/overrides/my_datamodule_overrides.yaml` like below

    .. code-block:: yaml
    
        setup: # name of function to override
          _target_: src.utils.hydra.partial # leverage partial
          _partial_: src.utils.hydra.setup_my_dataset # path to function

3. Add the override to your model or datamodule like, for instance, in `/configs/datamodules/my_datamodule.yaml`:     

    .. code-block:: yaml
         
        defaults:
        - /overrides: my_datamodule

The most common use cases to override existing functions are:

a. Provide your own datamodule for :obj:`src.datamodules.base.BaseDataModule`
b. Override existing or add functions to :obj:`src.modules.base.TridentModule`

.. _mixins:

Mixins
------

In the context of trident, mixins constitute a series of methods that define behaviour of your model oder datamodule. Your mixins must not reinstantiate `:obj:pytorch_lightning.{LightningModule, LightningDataModule}`, but instead follow the below pattern.


.. code-block:: python

    class MyModelMixin:

        def __init__(self) -> None:
            # self.hparams comprises the instantiated attributes
            self.my_object = hydra.utils.instantiate(self.hparams.my_new_attribute)
        
        # override model forward
        def forward(self, batch: BatchEncoding) -> BaseModelOuput:
            ...

        # add new functions
        def other_function1(self, *args, **kwargs) -> Any:
            ...

        def other_function2(self, *args, **kwargs) -> Any:
            ...

You then provide paths to the objects in `/configs/mixins`:

:obj:`/configs/mixins/my_model_mixin.yaml`

.. code-block:: yaml

   - src.my_modules.mixin.MyModelMixin
    

.. _evaluation:

Evaluation
----------

The evaluation mixin diminishes the boilerplate when writing custom evaluation loops for custom models. The below example is an annotated variant of :repo:`sequence classification <configs/evaluation/sequence_classification.yaml>` (see also, :repo:`tatoeba <configs/evaluation/tatoeba.yaml>` for sentence translation retrieval).

The configuration separates on a high level into:

* **apply**: transformation functions applied to `batch`, `outputs`, and `step_outputs`
* **step_outputs**: what keys of (default: complete `batch` and `outputs`)
* **metric**: configure how to instantiate and compute your metric

.. code-block:: yaml

    # apply transformation function 
    apply:
      batch: null
      outputs:   
        _target_: src.utils.hydra.partial
        _partial_: src.evaluation.classification.get_preds

      step_outputs: null  # on flattened outputs of what's collected from steps

    # Which keys/attributes are supposed to be collected from `outputs` and `batch`
    # for {val, test} loop end
    step_outputs:
      outputs: "preds" # can be a str
      batch: # or a list[str]
        - labels

    # metrics config
    metrics:
      # name of the metric used eg for logging
      accuracy:
        # instructions to instantiate metric, preferrably torchmetrics.Metric
        metric:
          _target_: torchmetrics.Accuracy

        # either on_step: true or on_epoch: true
        on_step: true # torchmetrics compute on_step!

        # either on_step: true or on_epoch: true
        compute: 
          # function_argument: "from:key"
          # ... for `preds` of `torchmetrics.Accuracy` get `preds` from `outputs`
          preds: "outputs:preds"
          # ... for `targets` of `torchmetrics.Accuracy` get `labels` from `batch`
          target: "batch:labels"

      f1:
        metric:
          _target_: torchmetrics.F1
        on_step: true
        compute:
          preds: "outputs:preds"
          target: "batch:labels"


where `get_preds` is defined as follows: 

.. code-block:: python
    
    def get_preds(outputs):
        outputs.preds = outputs.logits.argmax(dim=-1)
        return outputs
