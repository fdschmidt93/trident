#######################
|project| in 20 minutes
#######################

The walkthrough first introduces common concepts of hydra_ and then walks through an exemplary text-classification pipeline for sequence-pair classification (NLI).

hydra primer
============

It is important to have basic familiarity with hydra_, which shines at bottom-up hierarchical yaml configuration. A key feature of hydra_ for |project| is the `defaults-list <https://hydra.cc/docs/advanced/defaults_list/>`_ . Below is a brief primer of hydra_ for its use in |project|.

General
-------

- ``_target_: transformers.AutoModel.from_pretrained``: the ``__target__`` points to the Python function / method that initializes the object
- ``_recursive_: false`` ensures that objects are not instantiated eagerly, but only when instantiated explicitly. |project| takes care of instantiating your objects at the right time to bypass hydra_ limitations
- ``_partial_: true`` is common to instantiate functions with pre-set arguments and keywords

Bottom-Up Hierarchical Configuration
------------------------------------

- By directory, ``default.yaml`` defines defaults or required attributes (``???``) for the corresponding component (e.g., ``module``, ``datamodule``)
- Sub-directories *can* define the corresponding sub-configuration, like ``configs/datamodule/dataloaders.yaml`` is inherited in the ``defaults`` list of ``datamodule`` in ``configs/datamodule/default.yaml``.

  .. code-block:: yaml
    
      # configs/datamodule/default.yaml
      defaults:
        # pull in configs/datamodule/dataloaders/default.yaml
        - dataloaders: default
        - datasets: null

Project Structure
=================

An exemplary structure for a user project is shown below:

- `configs <https://github.com/fdschmidt93/trident/tree/main/examples/configs>`_ holds the entire hydra_ yaml configuration
-`src <https://github.com/fdschmidt93/trident/tree/main/examples/src>`_ comprises required code, typically for processing and evaluation, as referred to in the config

.. code-block::

    # yaml configuration
    your-project
    ├── configs
    │   ├── config.yaml # inherits all `default.yaml`
    │   ├── experiment # typical entry point, denotes 2nd-level `config.yaml` for your experiment
    │   │   ├── default.yaml
    │   │   └── nli.yaml
    │   ├── module
    │   │   ├── default.yaml
    │   │   ├── evaluation
    │   │   │   └── text_classification.yaml
    │   │   ├── optimizer
    │   │   │   ├── adam.yaml
    │   │   │   └── adamw.yaml
    │   │   ├── scheduler
    │   │   │   └── linear_warm_up.yaml
    │   │   └── text_classification.yaml
    │   ├── datamodule
    │   │   ├── default.yaml
    │   │   ├── text_classification.yaml
    │   │   └── nli.yaml
    │   ├── dataspec
    │   │   ├── dataloader
    │   │   │   └── default.yaml
    │   │   ├── evaluation
    │   │   │   └── text_classification.yaml
    │   │   ├── default.yaml
    │   │   ├── text_classification.yaml
    │   │   ├── nli.yaml
    │   │   ├── xnli_val_test.yaml
    │   │   └── amnli_val_test.yaml
    │   ├── hydra
    │   │   └── default.yaml
    │   ├── logger
    │   │   ├── csv.yaml
    │   │   └── wandb.yaml
    │   ├── callbacks
    │   │   └── default.yaml                 
    │   └── trainer
    │       ├── debug.yaml
    │       └── default.yaml
    └── src
        └── tasks
            └── text_classification
                ├── evaluation.py
                └── processing.py

Components
==========

TridentModule
-------------

:class:`~trident.core.module.TridentModule` extends the LightningModule_. The configuration defines all required components for a :class:`~trident.core.module.TridentModule`:

1. ``model``: ``__target__`` to your model constructor for which ``TridentModule.model`` will be initialized
2. ``optimizer``: the optimizer for all :class:`~trident.core.module.TridentModule` parameters
3. ``scheduler``: the learning-rate scheduler for the ``optimizer``

The ``default.yaml`` by default sets up AdamW optimizer and linear learning rate scheduler.

.. code-block:: yaml

    # _target_ is hydra-lingo to point to the object (class, function) to instantiate
    _target_: trident.TridentModule
    # _recursive_: true would mean all keyword arguments are /already/ instantiated
    # when passed to `TridentModule`
    _recursive_: false

    defaults:
    # interleaved with setup so instantiated later (recursive false)
    - optimizer: adamw.yaml  # see config/module/optimizer/adamw.yaml for default
    - scheduler: linear_warm_up  # see config/module/scheduler/linear_warm_up.yaml for default
    
    # required to be defined by user
    model: ???

A common pattern is that users create a ``configs/module/task.yaml`` that predefines shared ``model`` and ``evaluation`` logic for a particular task.

.. code-block:: yaml

    defaults:
      - default
      - evaluation: text_classification
    model:
      _target_: transformers.AutoModelForSequenceClassification.from_pretrained
      num_labels: ???
      pretrained_model_name_or_path: ???

- The ``model`` constructor points to ``transformers.AutoModelForSequenceClassification.from_pretrained``.
  The actual model and number of labels will be defined in either the experiment configuration or in the CLI (cf. ``???``).


TridentDataspec
---------------

A :class:`~trident.core.dataspec.TridentDataspec` class encapsulates the configuration for data handling in a machine learning workflow. It manages various aspects of data processing including dataset instantiation, preprocessing, dataloading, and evaluation.

**Configuration Keys**

- ``dataset``: Specifies how the dataset should be instantiated.
- ``dataloader``: Defines the instantiation of the ``DataLoader``.
- ``preprocessing`` (optional): Details the methods or function calls for dataset preprocessing.
- ``evaluation`` (optional): Outlines any post-processing steps and metrics for dataset evaluation.
- ``misc`` (optional): Reserved for miscellaneous settings that do not fit under other keys.

.. _preprocessing:

preprocessing
^^^^^^^^^^^^^

The ``preprocessing`` key in the configuration details the steps for preparing the dataset. It includes two special keys, ``method`` and ``apply``, each holding dictionaries for specific preprocessing actions.

- ``method``: Contains dictionaries of class methods along with their keyword arguments. These are typically methods of the dataset class.
- ``apply``: Comprises dictionaries of user-defined functions, along with their keyword arguments, to be applied to the dataset.

The preprocessing fucntions take the ``Dataset`` as the first positional argument. The functions are called in order of the configuration. Note that ``"method"`` is a convenience keyword which can also be achieved by pointing to the classmethod in ``"_target_"`` of an ``"apply"`` function.

**Example Configuration**

.. code-block:: yaml

    preprocessing:
      method:
        map: # dataset.map of huggingface `datasets.arrow_dataset.Dataset`
          function:
            _target_: src.tasks.text_classification.processing.preprocess_fn
            _partial_: true
            column_names:
              text: premise
              text_pair: hypothesis
            tokenizer:
              _partial_: true
              _target_: transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
              self:
                  _target_: transformers.AutoTokenizer.from_pretrained
                  pretrained_model_name_or_path: ${module.model.pretrained_model_name_or_path}
              padding: false
              truncation: true
              max_length: 128
        # unify output format of MNLI and XNLI
        set_format:
          columns:
            - "input_ids"
            - "attention_mask"
            - "label"

dataloader
^^^^^^^^^^

The DataLoader configuration (`configs/dataspec/dataloader/default.yaml`) is preset with reasonable defaults, accommodating typical use cases.

**Example Configuration**

.. code-block:: yaml

    _target_: torch.utils.data.dataloader.DataLoader
    collate_fn:
      _target_: transformers.data.data_collator.DataCollatorWithPadding
      tokenizer:
        _target_: transformers.AutoTokenizer.from_pretrained
        pretrained_model_name_or_path: ${module.model.pretrained_model_name_or_path}
      max_length: ???
    batch_size: 32
    pin_memory: true
    shuffle: false
    num_workers: 4


.. _evaluation:

evaluation
^^^^^^^^^^

The logic of evaluation is defined in ``./configs/dataspec/evaluation/text_classification.yaml``. It is common to define evaluation per type of task.

``evaluation`` configuration segments into the fields ``prepare``, ``step_outputs``, and ``metrics``.

.. seealso:: :py:class:`trident.utils.types.EvaluationDict`


prepare
"""""""

``prepare`` defines functions called on the ``batch``, the model ``outputs``, or the collected ``step_outputs``.

The :class:`~trident.core.module.TridentModule` hands the below keywords to facilitate evaluation. Since the :class:`~trident.core.module.TridentModule` extends the LightningModule_, useful attributes like ``trainer`` and ``trainer.datamodule`` are available at runtime.

**Example Configuration**

.. code-block:: yaml

    prepare:
      # takes (trident_module: TridentModule, batch: dict, split: Split) -> dict
      batch: null            
      # takes (trident_module: TridentModule, outputs: dict, batch: dict, split: Split) -> dict
      outputs:
        _partial_: true
        _target_: src.tasks.text_classification.evaluation.get_preds
      # takes (trident_module: TridentModule, step_outputs: dict, split: Split) -> dict
      step_outputs: null     

where ``get_preds`` is defined as follows and merely adds  

.. code-block:: python
    
    def get_preds(outputs: dict, *args, **kwargs) -> dict:
        outputs["preds"] = outputs["logits"].argmax(dim=-1)
        return outputs

.. seealso:: :py:class:`trident.utils.enums.Split`, :py:class:`trident.utils.types.PrepareDict`

step_outputs
""""""""""""

``step_outputs`` defines what keys are collected from a ``batch`` or ``outputs`` dictionary, per step, into the flattened outputs ``dict`` per evaluation dataloader. The flattened dictionary then holds the corresponding key-value pairs as input to the ``prepare_step_outputs`` function, which ultimately serves at input to metrics computed at the end of an evaluation loop.

.. note:: |project| ensures that after each evaluation loop, lists of ``np.ndarray``\s ``torch.Tensor``\s are correctly stacked to single array with appropriate dimensions.

**Example Configuration**

.. code-block:: yaml

    # Which keys/attributes are supposed to be collected from `outputs` and `batch`
    step_outputs:
      # can be a str
      batch: labels
      # or a list[str]
      outputs:
        - "preds"
        - "logits"

.. seealso:: :py:function:`trident.utils.flatten_dict`

metrics
"""""""

``metrics`` denotes a dictionary for all evaluated metrics. For instance, a metric such as ``acc`` may contain:

- ``metric``: how to instantiate the metric; typically a ``partial`` function; must return a ``Callable``.
- ``compute_on``: Either ``eval_step`` or ``epoch_end``, with the latter being the default.
- ``kwargs``: A custom syntax to fetch ``kwargs`` of ``metric`` from one of the following: ``[trident_module, outputs, batch, cfg]``.
  - ``outputs`` refers to the model ``outputs`` when ``compute_on`` is set to ``eval_step`` and to ``step_outputs`` when ``compute_on`` is set to ``epoch_end``.

In the NLI example:
  - The keyword ``preds`` for ``torchmetrics.functional.accuracy`` is sourced from ``outputs["preds"]``.
  - The keyword ``target`` for ``torchmetrics.functional.accuracy`` is sourced from ``outputs["labels"]``.

**Example Configuration**

.. code-block:: yaml

    metrics:
      # name of the metric used eg for logging
      acc:
        # instructions to instantiate metric, preferrably torchmetrics.Metric
        metric:
          _partial_: true
          _target_: torchmetrics.functional.accuracy
        # either "eval_step" or "epoch_end", defaults to "epoch_end"
        compute_on: "epoch_end"
        kwargs: 
          preds: "outputs:preds"
          target: "outputs:labels"


TridentDataModule
-----------------

The default configuration (``configs/datamodule/default.yaml``) for a :class:`~trident.core.datamodule.TridentDatamodule` defines how training and evaluation datasets are instantiated.
Each split is a dictionary of :class:`~trident.core.dataspec.TridentDataspec`.

.. code-block:: yaml

    _target_: trident.TridentDataModule
    _recursive_: false

    misc:
        # reserved key for general TridentDataModule configuration
    train:
        # DictConfig of TridentDataspec
    val:
        # DictConfig of TridentDataspec
    test:
        # DictConfig of TridentDataspec

Config Composition
^^^^^^^^^^^^^^^^^^

.. note:: Hierarchical config composition heavily relies on `default lists <https://hydra.cc/docs/advanced/defaults_list/>`_ .

The below file tree is a common structure for a hierarchical :class:`~trident.core.datamodule.TridentDatamodule` configuration in our NLI example.

We will hierarchically

1. Compose a general ``dataspec``
2. Compose a tast-specific text classification ``dataspec``
3. Compose a NLI ``dataspec``
4. Compose a dictionary of NLI ``dataspec``s
5. Compose a datamodule

.. code-block:: bash

    configs
    ├── config.yaml
    ├── datamodule
    │   ├── amnli_val_test.yaml
    │   ├── default.yaml
    │   ├── mnli_train.yaml
    │   └── xnli_val_test.yaml
    └── dataspec
        ├── dataloader
        │   └── default.yaml
        ├── evaluation
        │   └── text_classification.yaml
        ├── default.yaml
        ├── nli.yaml
        ├── text_classification.yaml
        ├── xnli_val_test.yaml
        └── amnli_val_test.yaml

Default
"""""""

The general ``dataspec`` simply defines the default (``./configs/dataspec/default.yaml``) configuration.

.. code-block:: yaml

    defaults:
      - dataset: null
      # pull in the default dataloader
      - dataloader: default

    dataset:
      #
      _target_: datasets.load.load_dataset

Text Classification
"""""""""""""""""""

.. code-block:: yaml

    defaults:
      - default
      - evaluation: text_classification # see TridentDataspec evaluation
    
    # task specific preprocessing
    preprocessing:
        ... # see TridentDataspec preprocessing


.. seealso::
    :ref:`TridentDataspec.preprocessing <preprocessing>`, :ref:`TridentDataspec.preprocessing <preprocessing>`

NLI
"""

The ``configs/dataspec/nli.yaml`` simply extends the task-specific ``text_classification.yaml`` by specifying columns for the tokenizer in preprocessing.

.. code-block:: yaml

    defaults:
      - text_classification

    preprocessing:
      map:
        function:
          column_names:
            text: premise
            text_pair: hypothesis

.. seealso::
    :ref:`TridentDataspec.preprocessing <preprocessing>`, :ref:`TridentDataspec.preprocessing <preprocessing>`

Dictionary of NLI
"""""""""""""""""

The ``configs/dataspec/xnli_val_test.yaml`` levers ``hydra`` `package directives <https://hydra.cc/docs/advanced/overriding_packages/>`_ to put the ``nli`` configuration into the corresponding dataspec keys.

.. code-block:: yaml

    defaults:
      - nli@validation_xnli_en
      - nli@validation_xnli_es
      # ... can extend this to the entire XNLI benchmark for val and test splits
    validation_xnli_en:
      dataset:
        path: xnli
        name: en
        split: validation
    validation_xnli_es:
      dataset:
        path: xnli
        name: es
        split: validation
    # ... can extend this to the entire XNLI benchmark for val and test splits

Optional NLI datamodules
""""""""""""""""""""""""

Experiment
----------

The experiment configurations also segments into a general ``default.yaml`` and a task-specific ``nli.yaml``.


.. code-block:: yaml

    defaults:
      - override /trainer: default
      - override /callbacks: default
      - override /config_callbacks: default
      - override /logger: wandb

    seed: 42
    task: nli

    trainer:
      max_epochs: 10
      devices: 1
      precision: "16-mixed"
      deterministic: true
      inference_mode: false

    logger:
      wandb:
        name: "model=${module.model.pretrained_model_name_or_path}_epochs=${trainer.max_epochs}_bs=${oc.select:datamodule.dataloaders.train.batch_size, ${datamodule.dataloaders.batch_size}}_lr=${module.optimizer.lr}_scheduler=${module.scheduler.num_warmup_steps}_seed=${seed}"
        tags:
          - "${module.model.pretrained_model_name_or_path}"
          - "bs=${oc.select:datamodule.dataloaders.train.batch_size, ${datamodule.dataloaders.batch_size}}"
          - "lr=${module.optimizer.lr}"
          - "scheduler=${module.scheduler.num_warmup_steps}"
        project: ${task}

Commandline Interface
=====================

hydra_ allows to simply set configuration items on the commandline. See more information

.. code-block:: bash

    # change the learning rate
    python -m trident.run experiment=nli module.optimizer.lr=0.0001
    # set a different optimizer
    python -m trident.run experiment=nli module.optimizer=adam
    # no lr scheduler
    python -m trident.run experiment=nli module.scheduler=null
