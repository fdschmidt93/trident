# Changelog

## 0.1.0 (2024-05-21)


### ⚠ BREAKING CHANGES

* **dataset:** merge get and get_raw to get
* **evaluation:** pass split as enum to prepare functions
* **eval:** consistent multi-datasets order in prepare cfg

### Features

* allow shared sub-key config in datasets ([a3941d8](https://github.com/fdschmidt93/trident/commit/a3941d895bfcebe0c7408c8e2c54c19d5e61a40b))
* check num_samples on eval_epoch_end ([e56c8a3](https://github.com/fdschmidt93/trident/commit/e56c8a3bdd58569b4b743b47f92b6d0a79c5e354))
* **ci:** add release work flow ([4c8369a](https://github.com/fdschmidt93/trident/commit/4c8369a0131532f641986beed19534783302826e))
* **config:** ensure dataset keys are coherent ([28473b1](https://github.com/fdschmidt93/trident/commit/28473b1c27b2ec2418aa7bfb1853e456a7875358))
* **datamodule:** update spec ([da87617](https://github.com/fdschmidt93/trident/commit/da8761707d2f21d5e5560973f1cf6100efc1f815))
* docgen ci ([7984585](https://github.com/fdschmidt93/trident/commit/7984585ab73abb83b3d2ebea41716db9741e641e))
* **docs:** add NLI example repository ([042c5b1](https://github.com/fdschmidt93/trident/commit/042c5b1fd4fad12bc1d20d18c73ad8523957704e))
* **docs:** cleanly add example not as submodule ([c214dfa](https://github.com/fdschmidt93/trident/commit/c214dfaf0bac840355feac506d6ebe23b321defa))
* **docs:** extended hydra primer ([05d626a](https://github.com/fdschmidt93/trident/commit/05d626a0fb4b69ea63a4419a206c9696296f8862))
* **docs:** how to set variable & runtime dir ([207e801](https://github.com/fdschmidt93/trident/commit/207e8011ce9eda118db3bfb8dfefbd59e63ba477))
* **docs:** improve QA ([687d9ff](https://github.com/fdschmidt93/trident/commit/687d9ff584584d2600e551226185590a7c0d9129))
* **docs:** overhaul documentation ([a055ebe](https://github.com/fdschmidt93/trident/commit/a055ebe3fb769787a03e7e45e788ecc1e4de5fa9))
* **docs:** refer to example project repo in walkthrough ([2ce6756](https://github.com/fdschmidt93/trident/commit/2ce6756edb52554e1bd51d9cd1654a1b0bb469ed))
* **eval:** consistent multi-datasets order in prepare cfg ([bb2d8a4](https://github.com/fdschmidt93/trident/commit/bb2d8a4690592d9f1f7eb1b13a83b6e94472506f))
* **evaluation.step_outputs:** support * wildcards ([797f89f](https://github.com/fdschmidt93/trident/commit/797f89f2a2e5e11f206b7c4f7dd5d147104c00e2))
* **evaluation:** handle non-list primitives better ([d4fc521](https://github.com/fdschmidt93/trident/commit/d4fc521cd8523e86a178e850570e6b31703ba0d4))
* **evaluation:** pass split as enum to prepare functions ([37a3d4a](https://github.com/fdschmidt93/trident/commit/37a3d4a2209a630cfa6acd7ea0b64c38b5617548))
* **evaluation:** proper multi-GPU validation ([cfbf557](https://github.com/fdschmidt93/trident/commit/cfbf5579ce581fbc58d76aff02ae602b177d19db))
* extend dm, model, evalmixin ([e64296c](https://github.com/fdschmidt93/trident/commit/e64296cf10d12aa23c3d1becb71abd105119e8fe))
* **hydra:** test hierarchical config management ([dfe9aff](https://github.com/fdschmidt93/trident/commit/dfe9affa027998d6e565f62898b2024a3991cad0))
* load checkpoints via run.ckpt_path ([6a3fc6f](https://github.com/fdschmidt93/trident/commit/6a3fc6fec7e057dc9e4abd1434be32165ad06b90))
* **logging:** add `dataset_name` & `split` to metric fields ([7828e4d](https://github.com/fdschmidt93/trident/commit/7828e4d5774b6260dab224ae44efe7075764b460))
* **logging:** bypass logging (e.g. storing predictions) ([852d58e](https://github.com/fdschmidt93/trident/commit/852d58e52cd8af56b38d3f042fc85f36269f3703))
* more datamodules & start to revamp evalmixin ([7cb30aa](https://github.com/fdschmidt93/trident/commit/7cb30aa5deaf127394e7c01243a7fb29c449e88c))
* progress towards 0.1 again ([62a24bc](https://github.com/fdschmidt93/trident/commit/62a24bc5d34a6e84211e67464532ad24d9f86f40))
* **run:** enable bypassing trainer.fit ([2047284](https://github.com/fdschmidt93/trident/commit/2047284caaaa36dc95bbab86ab2bf2d79df35b61))
* **run:** generalize overriding train.py ([c3375b5](https://github.com/fdschmidt93/trident/commit/c3375b59b80748ba7e0ec952d108eee0c5aa9066))
* **run:** upstream execution to trident-core ([136c1f7](https://github.com/fdschmidt93/trident/commit/136c1f7771c2e75142d49a455efe4509db3630cf))
* setup gh pages ([8e713bf](https://github.com/fdschmidt93/trident/commit/8e713bf5e1829bfd9d793e0491c13703f8291f25))
* **test:** add simple multi train dataset test ([#9](https://github.com/fdschmidt93/trident/issues/9)) ([0147a9d](https://github.com/fdschmidt93/trident/commit/0147a9dda06551656d2b09a1d69764d1ac00425d))
* **tests:** add CI ([436fe6f](https://github.com/fdschmidt93/trident/commit/436fe6f83c41fa6ba423663591e2edc664eac4d2))
* **tests:** add many explict eval config tests ([7fdc297](https://github.com/fdschmidt93/trident/commit/7fdc297cd18a9ba04deae5546818c3938f9cc3c2))
* **tests:** add simple tests ([afd0950](https://github.com/fdschmidt93/trident/commit/afd0950075da860ec734e3e4af7e9c04ea503062))
* **tests:** add tests for single & multi datasets ([8accc91](https://github.com/fdschmidt93/trident/commit/8accc914ad9288bce899d3dfe25d6a23bbde0303))
* **tests:** test LR Scheduler with accumulation ([2b2e74d](https://github.com/fdschmidt93/trident/commit/2b2e74d19fae121f7ef180b7cdb42cf944bf597b))
* **transform:** handling of 0d & 3d tensors ([22b6842](https://github.com/fdschmidt93/trident/commit/22b6842da8f01c9b4933b4ab14d2650275355e48))
* **utils:** improve and test `expand` ([97f9945](https://github.com/fdschmidt93/trident/commit/97f9945ae58ba1047fd770a7b36e1e5b5e3bff37))
* **utils:** make instantiate_and_apply more robust and add tests ([f916937](https://github.com/fdschmidt93/trident/commit/f916937a5e9445c09ebb55d3de7e7873c77d64c8))


### Bug Fixes

* **cfg:** use `run` for run-specific variables ([89655d7](https://github.com/fdschmidt93/trident/commit/89655d7fc9dd50f4d2bc53bfb05d16b344936732))
* **CI:** dependencies ([65e5df3](https://github.com/fdschmidt93/trident/commit/65e5df3fb85826c7842c8c78e4a812f226a48fc9))
* datamodule ([2ea6ced](https://github.com/fdschmidt93/trident/commit/2ea6ced0345f924ebba7e30738d952a78237599b))
* **datamodule:** __iter__ for combined train loader ([46c9652](https://github.com/fdschmidt93/trident/commit/46c9652fed5b90480ec560f339ce0751f985ec24))
* **datamodule:** default trident configuration ([60c62db](https://github.com/fdschmidt93/trident/commit/60c62db5217d4d203a5a16f10c4a7ec544480a88))
* **datamodule:** getting train dataset length ([09dc5ff](https://github.com/fdschmidt93/trident/commit/09dc5ff3a09b3575d0829b74f40b4199dff9992a))
* docs ([7b0bc77](https://github.com/fdschmidt93/trident/commit/7b0bc77133b9fef3deb82177062ce42af8b1cf4c))
* **docs:** add dataset_name to signature of prepare functions ([c2bcc62](https://github.com/fdschmidt93/trident/commit/c2bcc62e3dad7a75e9c83c764c64710c4efc7c5f))
* **docs:** quotation for runtime directory ([74c66d1](https://github.com/fdschmidt93/trident/commit/74c66d1f1d7326c104b1d36ba4dba77bfa90f7f6))
* **evaluation:** correctly pass dataset_name to prepare functions ([81d1023](https://github.com/fdschmidt93/trident/commit/81d10230f79a950bbdc2ec02951908e8f3e84fd2))
* **evaluation:** typo in check for step_collection_dico with datasets ([#7](https://github.com/fdschmidt93/trident/issues/7)) ([be9b33e](https://github.com/fdschmidt93/trident/commit/be9b33edac58f7a5006e1a0d18c7b9de3f5ba0cf))
* **examples:** add missing indicxnli dataspecs ([31e4306](https://github.com/fdschmidt93/trident/commit/31e43065919cafc7ee2ce0716a03d873aa275372))
* gradient accumulation taken care of by lightning ([3b46f39](https://github.com/fdschmidt93/trident/commit/3b46f399493cbfbeadbaaf2c1a74cf20dee15397))
* **hydra:** handling of non-primitives in _method_ ([2e8c7dc](https://github.com/fdschmidt93/trident/commit/2e8c7dc215787317964fa1c718f5e35f73463262))
* **hydra:** resolve parent-level links in config expansion ([498a75c](https://github.com/fdschmidt93/trident/commit/498a75c611435f520838fc4d56f0d4c824c77d50))
* include readme in index ([0703a17](https://github.com/fdschmidt93/trident/commit/0703a17ebb3602480c0118daf2059513e96ea1d1))
* **logging:** correct which keys to log ([d789c7c](https://github.com/fdschmidt93/trident/commit/d789c7c1e1da03a3a4df7ee7f327b993e5b8382d))
* **logging:** resolve referenced values ([a3a7483](https://github.com/fdschmidt93/trident/commit/a3a7483331c4c9e44707f3ab30c7b2fad4a64506))
* **misc:** fixes from prior update ([823db71](https://github.com/fdschmidt93/trident/commit/823db7126111d84848a24e2edfde845d82a05751))
* missing default argument for module_cfg ([eeee5a3](https://github.com/fdschmidt93/trident/commit/eeee5a38bd41852102b8dd23c55fac33d54ac324))
* multiple eval dataloaders ([5afaf70](https://github.com/fdschmidt93/trident/commit/5afaf702a33a825ce5c56f511514f03efe2949d1))
* only trigger docgen for push to main ([a8c4e35](https://github.com/fdschmidt93/trident/commit/a8c4e35d54d863ad6873cba6ba9ac49c17650458))
* **optimizer:** assert type of trainer.max_epochs ([7c75de9](https://github.com/fdschmidt93/trident/commit/7c75de9765bf15eaf05feb90405364d65d7b69f4))
* **optimizer:** correct order of imports ([c2fd4f9](https://github.com/fdschmidt93/trident/commit/c2fd4f9f9465609e45adc90100ab0a0298fa735c))
* **optimizer:** int as valid num_warmup_steps ([134a041](https://github.com/fdschmidt93/trident/commit/134a04188a5269929883632a0056ac654f98b181))
* **optimizer:** scheduler with accumulate_grad_batches &gt; 1 ([af2b333](https://github.com/fdschmidt93/trident/commit/af2b333592fa7643474e03a43ecf5b2aa9898a56))
* **run:** default limit batches ([b2df173](https://github.com/fdschmidt93/trident/commit/b2df173a7163f4edc3941763ba2c9bdaeedcea71))
* **run:** model checkpoint check ([3534c3c](https://github.com/fdschmidt93/trident/commit/3534c3cb72665cc422e8c7e192109c23dd4b68ce))
* **run:** remove test_after_training ([a45c003](https://github.com/fdschmidt93/trident/commit/a45c00386ae9b1f3e649c7448a58e5609f964b3b))
* **run:** skip/do testing correctly ([ca137d7](https://github.com/fdschmidt93/trident/commit/ca137d7a0000d39f30822abe8900ae987ce8c0cf))
* setup ([a91a10d](https://github.com/fdschmidt93/trident/commit/a91a10da11b2ae1a1880de760d2625e1e25e4661))
* setup ([a18c24b](https://github.com/fdschmidt93/trident/commit/a18c24b31f2e72feb54a62883c24eb11700b2c1a))
* styling of docs on gh-pages ([ede6d71](https://github.com/fdschmidt93/trident/commit/ede6d71489d8d74a7b6741f7710ac490e889f03b))
* styling of docs on gh-pages ([7f0a9c6](https://github.com/fdschmidt93/trident/commit/7f0a9c6325b6092326596077f5d1502507f3fb91))
* **train:** safeguard seed access ([668df26](https://github.com/fdschmidt93/trident/commit/668df2616ae6bc03318c4a983c30560db2af377d))
* **types:** annotation for dataloader getters ([6c7567f](https://github.com/fdschmidt93/trident/commit/6c7567f92fdfc7faef24899aa17ec4aa466126d9))


### Documentation

* **contributing:** remove stale paragraph ([6abe64c](https://github.com/fdschmidt93/trident/commit/6abe64ce697f6579e7566db09af2cc068df72b59))
* document `_self_` in defaults list ([2950434](https://github.com/fdschmidt93/trident/commit/29504346845808698ff46d3cc4074dde822ad39b))
* multi-GPU training on IterableDataset ([afdcb56](https://github.com/fdschmidt93/trident/commit/afdcb56941564012358c04df6ebd649147d9e113))
* **QA:** how to bypass training ([11e4437](https://github.com/fdschmidt93/trident/commit/11e4437151033df3631888b41618ba540f73616a))


### Miscellaneous Chores

* **dataset:** merge get and get_raw to get ([fcab324](https://github.com/fdschmidt93/trident/commit/fcab32405f0861ffe4aa0bf2a41cb5af7480e2f9))
