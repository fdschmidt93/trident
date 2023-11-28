
It is important to have basic familiarity with hydra_, which shines at bottom-up hierarchical yaml configuration. A key feature of hydra_ for |project| is the `defaults-list <https://hydra.cc/docs/advanced/defaults_list/>`_ . Below is a brief primer of hydra_ for its use in |project|.

Foreword
--------

In the context of |project|, we will use hydra_ to declare our 

- Trainer_: declares how training, validation, and testing is run
- :class:`~trident.core.module.TridentModule`: declares the your "model"
- :class:`~trident.core.datamodule.TridentDatamodule`: declares your ``train``, ``val``, and ``test`` splits

General
-------

hydra_ is a Python yaml framework to compose complex pipelines. The directory tree of your configuration in our simplified example looks like this.

.. code-block:: bash

    configs
    ├── config.yaml
    ├── datamodule
    │   ├── default.yaml
    │   └── my_experiment.yaml
    ├── datamodule
    │   ├── default.yaml
    │   └── my_datamodule.yaml
    └── module
        ├── default.yaml
        └── my_module.yaml

Structure
^^^^^^^^^

A certain set of principles are set into hydra_

- The configuration relies on relative paths in the configuration folder structure

Special Keywords
^^^^^^^^^^^^^^^^

.. list-table:: Title
   :widths: 25 10 65
   :header-rows: 1

   * - **Key**
     - **Type*
     - **Description**

   * - ``_target_``
     - ``str``
     - The ``_target_`` points to the Python function / method that initializes the object

   * - ``_recursive_``
     - ``bool``
     - Do not eagerly instantiate sub-keys that comprises ``_target_``, but only when ``hydra.utils.instantiate`` is called

   * - ``_partial_``
     - ``bool``
     - Instantiate ``_target_`` function curried with pre-set arguments and keywords

   * - ``_args_``
     - ``list[Any]``
     - Positional arguments for the ``__target__`` at ``hydra.utils.instantiate``


Object and Function Instantiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

hydra_ allow you to define Python objects and functions declaratively in yaml. The below example instantiates an ``AutoModelForSequenceClassification`` with ``roberta-base``.

.. code-block:: yaml

   sequence_classification_model:
       # path to the object we want to instantiate
       _target_: transformers.AutoModelForSequenceClassification
       # kwargs for the object we want to instantiate
       # can be themselves instantiated!
       pretrained_model_name_or_path: "roberta-base"
       num_labels: 3
       # optionally positional arguments
       # _args_:
       #  - 0
       #  - 1

.. note:: Carefully check whether the parent node of ``sequence_classification_model`` has ``_recursive_: true`` or not! If it is set to ``false``, the ``sequence_classification_model`` has to be instaniated manually (i.e., ``hydra.utils.instantiate(cfg.{...}.sequence_classification_model)``

hydra_ yaml configuration comprises various reserved keys:

- ``_target_: transformers.AutoModel.from_pretrained``: 
- ``_recursive_: false`` ensures that objects are not instantiated eagerly, but only when instantiated explicitly. |project| takes care of instantiating your objects at the right time to bypass hydra_ limitations
- ``_partial_: true`` is common to instantiate functions with pre-set arguments and keywords

defaults-list: Bottom-Up Hierarchical Configuration
---------------------------------------------------

- By directory, ``default.yaml`` defines defaults or required attributes (``???``) for the corresponding component (e.g., ``module``, ``datamodule``)
- Sub-directories *can* define the corresponding sub-configuration, like ``configs/datamodule/dataloaders.yaml`` is inherited in the ``defaults`` list of ``datamodule`` in ``configs/datamodule/default.yaml``.

  .. code-block:: yaml
    
      # configs/datamodule/default.yaml
      defaults:
        # pull in configs/datamodule/dataloaders/default.yaml
        - dataloaders: default
        - datasets: null

packaging: abitrarily inherit configs
-------------------------------------
