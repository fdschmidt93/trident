##########################
Frequently Asked Questions
##########################

This document comprises a few tips and tricks to illustrate common concepts with ``trident``. Be sure to read the :ref:`walkthrough <walkthrough>`.

Configuration
=============

How do I set a default for variable in yaml?
--------------------------------------------

The yaml configuration of hydra_ bases on  OmegaConf_. OmegaConf_ has support for `built-in <https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html#built-in-resolvers>`_ and `custom <https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html#id9>`_ resolvers, which, among other things, let's you define a default for your variable that can otherwise not be resolved.

.. code-block:: yaml

    # absolute_path_to_node is a link to a node like e.g., `with_default` 
    with_default: "${oc.select:absolute_path_to_node,default_value}"

How do I best set a custom value for a variable in yaml?
--------------------------------------------------------

The best practice to setting variables you want to change between runs in your experiments is to add them to the ``run`` key of your ``experiment`` configuration.

**Important**: You must then link the corresponding configuration (like batch sizes in
``dataloader``) for your user variables.

.. code-block:: yaml

    # defaults ...
    run:
      seed: 42
      task: ???
      # examples
      train_batch_size: 32
      val_test_batch_size: 128
    # remaining configuration ...

Module and Models
=================

How to use your own model?
--------------------------

Using your own model typically follows on of two patterns.

The existing model already defines a training step.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HuggingFace models merge the Pytorch Lighting `forward <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.forward>`__
and `training_step <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.training_step>`__
function into a single ``forward`` function that also accepts ``labels``
as a ``kwargs``. The ``trident.TridentModule`` seamlessly passes the
batch through to ``self.model(**batch)`` in ``forward``.

In these cases, the below pattern suffices

.. code:: yaml

   module:
       model:
         _target_: transformers.AutoModelForSequenceClassification.from_pretrained
         num_labels: ???
         pretrained_model_name_or_path: ???

which is hierarchically top-to-bottom constructed from interleaving your
``experiment`` configuration

.. code:: yaml

   defaults:
     # ...
     - override /module: text_classification.yaml
     # ...

sourced from ``{text, token}_classification``

.. code:: yaml

   defaults:
     - trident
     - /evaluation: text_classification

   model:
     _target_: transformers.AutoModelForSequenceClassification.from_pretrained
     num_labels: ???
     pretrained_model_name_or_path: ???

which inherits ``trident`` (i.e. optimizer, scheduler) defaults.

.. code:: yaml

   # _target_ is hydra-lingo to point to the object (class, function) to instantiate
   _target_: trident.TridentModule
   # _recursive_: true would mean all keyword arguments are /already/ instantiated
   # when passed to `TridentDataModule`
   _recursive_: false

   defaults:
   # interleaved with setup so instantiated later (recursive false)
   - /optimizer: ${optimizer}  # see config/optimizer/adamw.yaml for default
   - /scheduler: ${scheduler}  # see config/scheduler/linear_warm_up.yaml for default

   evaluation: ???
   model: ???

The existing model does not define a training step but a forward step.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this scenario, the user implements a ``trident.TridentModule``

.. code:: python

   class MyModule(TridentModule):
       def __init__(
           self,
           my_variable: Any,
           *args,
           **kwargs,
       ):
           super().__init__(*args, **kwargs)
           self.my_variable = my_variable

       # INFO: this is not stricty required and shows default implementation
       def forward(self, batch: dict) -> dict:
           # #####################
           # override IF AND ONLY IF custom glue between model and module required
           # #####################
           return self.model(**batch)

       # the default training_step implementation inherited in MyModule(TridentModule)
       def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
           # #####################
           # custom logic here -- don't forget to add  logging!
           # #####################
           outputs = self(batch)  # calls above forward(self, batch)
           self.log("train/loss", outputs["loss"])
           return outputs

and links the module intermittently in his own ``module`` configuration

.. code:: yaml

   module:
       # src.projects is an exemplary path in trident_xtreme folder
       _target_: src.projects.my_project.my_module.MyModule
       defaults:
         - trident
         - /evaluation: ??? # required, task-dependent

       model:
         _target_: my_package.my_existing_model
         model_kwarg_1: ???
         model_kwarg_2: ???

The architecture does not exist yet.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two variants are most common:

1. Write a ``lightning.pytorch.LightningModule`` for the barebones
   architecture (i.e. defining ``forward`` pass, model setup) and a
   separate ``TridentModule`` embedding the former to enclose training
   logic (``training_step``)
2. Write a stand-alone ``TridentModule`` that implements both
   ``forward`` and ``training_step``

The idiomatic approach is (1) as it reflects a more common research-oriented workflow.

How to opt out of default model instantiation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can opt out of automatic model instantiation by passing
``initialize_model=False`` to the ``super.__init__()`` method.

Beware that you the have to instantiate the ``self.model`` yourself!
Furthermore, you may need to override ``TridentModule.forward``, for instance, if the model is not defined in ``self.model`` any longer.

.. code:: python

   class MyModule(TridentModule):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, initialize_model=False, **kwargs)
           self.model = hydra.utils.instantiate(self.hparams.model)

How to load a checkpoint for a TridentModule?
---------------------------------------------

There are two options levering functionality of Lightning.

First, you can pass a to the top-level config which gets passed to
`Trainer.fit <https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit>`__

.. code:: yaml

   # ./configs/$YOUR_EXPERIMENT.yaml
   ckpt_path: absolute_path_to_ckpt

Second, you can use the built-in
`LightningModule.load_from_checkpoint <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.load_from_checkpoint>`__.

.. code:: yaml

   # typically declared in your ./configs/$YOUR_EXPERIMENT.yaml
   module:
     _target_: TridentModule.load_from_checkpoint
     checkpoint_path: (Union[str, Path, IO]) – Path to checkpoint. This can also be a URL, or file-like object
     map_location: str # typically "cpu" or "cuda:0"
     strict: true # like torch.load 
     # other kwargs: any extra kwargs to init the model. Can also be used to override saved hyperparameter values.

DataModule
==========

How to train and evaluate on multiple datasets?
-----------------------------------------------

The below example illustrates training and evaluating NLI jointly on English and a ``${lang}`` of `AmericasNLI <https://huggingface.co/datasets/americas_nli>`__.

.. code-block:: yaml

    defaults:
      - /dataspec@datamodule.train: 
        - mnli_train
        - amnli_train

During training ``batch`` turns from ``dict[str, torch.Tensor]`` with, for instance, a structure common for HuggingFace

.. code:: python

   batch = {
       "input_ids": torch.LongTensor,
       "attention_mask": torch.LongTensor,
       "labels": torch.LongTensor,
   }

to ``dict[str, dict[str, torch.Tensor]]``, embedding the original batch now by dataset.

.. code:: python

   batch = {
       "source": {
           "input_ids": torch.LongTensor,
           "attention_mask": torch.LongTensor,
           "labels": torch.LongTensor,
       }
       "target": {
           "input_ids": torch.LongTensor,
           "attention_mask": torch.LongTensor,
           "labels": torch.LongTensor,
       }
   }

**Important**: this is not applicable for evaluation, as ``dict[str, DataLoader]`` up- or downsample to the largest or smallest dataset in the dictionary. During evaluation, the ``DataLoader`` for multiple validation or test datasets consequently are of ``list[DataLoader]`` in order of declaration in the ``yaml`` configuration.

How do I subsample a dataset?
---------------------------

.. code:: yaml

   # typically declared in your ./configs/datamodule/$YOUR_DATAMODULE.yaml
   train:
     my_dataset:
       preprocessing:
         method:
           shuffle:
             seed: ${run.seed}
           select:
             indices:
               _target_: builtins.range
               _args_:
                 - 0
                 # must be set by user
                 - ${run.num_shots}

How do I only run testing?
--------------------------

Bypassing training is implemented with the corresponding Lightning Trainer_ flag. You can write the following in your ``experiment.yaml`` 

.. code-block:: yaml

   trainer:
    limit_train_batches: 0.0

or pass

.. code-block:: bash
    
    python -m trident.run ... trainer.limit_train_batches=0.0

to the CLI.
