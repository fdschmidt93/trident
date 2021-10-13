# Status
    * Base model (incl. optimizer and evaluation mixins) and datamodule provided: **easily** and **fully** customizable without boilerplate
    * Basic documentation is setup
    * Backbone for sequence classification, sentence translation retrieval ("aligned retrieval") and training QA is finished

# Next Steps
    * More documentation: CI (auto docgen on Github) and more explict documentation
    * Verify adding own datamodule is just as easy
    * Full coverage of XTREME: datamodules and evaluation
        - QA (evaluation), NER, Dependency Parsing, Multiple Choice are outstanding
    * Own custom model: fusing mSBERT -- stress testing my own work ;)
    * You guys get in on it: using, commenting, ideally contributing
    * After XTREME: language modelling support

# Framework
    * Flatten model arguments
    * Provide modular pipeline for tasks with everything defined _except for_ model
    * spin `src` into module
