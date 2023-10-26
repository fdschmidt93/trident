# Changelog

## 0.1.0 (2023-10-26)


### ⚠ BREAKING CHANGES

* **dataset:** merge get and get_raw to get
* **evaluation:** pass split as enum to prepare functions
* **eval:** consistent multi-datasets order in prepare cfg

### Features

* allow shared sub-key config in datasets ([a3941d8](https://github.com/fdschmidt93/trident/commit/a3941d895bfcebe0c7408c8e2c54c19d5e61a40b))
* check num_samples on eval_epoch_end ([e56c8a3](https://github.com/fdschmidt93/trident/commit/e56c8a3bdd58569b4b743b47f92b6d0a79c5e354))
* **ci:** add release work flow ([1bd1c38](https://github.com/fdschmidt93/trident/commit/1bd1c38a962a562f7acfebc4f72fb0fb14dd7e32))
* **config:** ensure dataset keys are coherent ([28473b1](https://github.com/fdschmidt93/trident/commit/28473b1c27b2ec2418aa7bfb1853e456a7875358))
* docgen ci ([7984585](https://github.com/fdschmidt93/trident/commit/7984585ab73abb83b3d2ebea41716db9741e641e))
* **eval:** consistent multi-datasets order in prepare cfg ([bb2d8a4](https://github.com/fdschmidt93/trident/commit/bb2d8a4690592d9f1f7eb1b13a83b6e94472506f))
* **evaluation:** pass split as enum to prepare functions ([37a3d4a](https://github.com/fdschmidt93/trident/commit/37a3d4a2209a630cfa6acd7ea0b64c38b5617548))
* extend dm, model, evalmixin ([e64296c](https://github.com/fdschmidt93/trident/commit/e64296cf10d12aa23c3d1becb71abd105119e8fe))
* **hydra:** test hierarchical config management ([dfe9aff](https://github.com/fdschmidt93/trident/commit/dfe9affa027998d6e565f62898b2024a3991cad0))
* more datamodules & start to revamp evalmixin ([7cb30aa](https://github.com/fdschmidt93/trident/commit/7cb30aa5deaf127394e7c01243a7fb29c449e88c))
* progress towards 0.1 again ([62a24bc](https://github.com/fdschmidt93/trident/commit/62a24bc5d34a6e84211e67464532ad24d9f86f40))
* **run:** generalize overriding train.py ([c3375b5](https://github.com/fdschmidt93/trident/commit/c3375b59b80748ba7e0ec952d108eee0c5aa9066))
* **run:** upstream execution to trident-core ([136c1f7](https://github.com/fdschmidt93/trident/commit/136c1f7771c2e75142d49a455efe4509db3630cf))
* setup gh pages ([8e713bf](https://github.com/fdschmidt93/trident/commit/8e713bf5e1829bfd9d793e0491c13703f8291f25))
* **test:** add simple multi train dataset test ([#9](https://github.com/fdschmidt93/trident/issues/9)) ([0147a9d](https://github.com/fdschmidt93/trident/commit/0147a9dda06551656d2b09a1d69764d1ac00425d))
* **tests:** add CI ([436fe6f](https://github.com/fdschmidt93/trident/commit/436fe6f83c41fa6ba423663591e2edc664eac4d2))
* **tests:** add many explict eval config tests ([7fdc297](https://github.com/fdschmidt93/trident/commit/7fdc297cd18a9ba04deae5546818c3938f9cc3c2))
* **tests:** add simple tests ([afd0950](https://github.com/fdschmidt93/trident/commit/afd0950075da860ec734e3e4af7e9c04ea503062))
* **tests:** add tests for single & multi datasets ([8accc91](https://github.com/fdschmidt93/trident/commit/8accc914ad9288bce899d3dfe25d6a23bbde0303))
* **utils:** improve and test `expand` ([97f9945](https://github.com/fdschmidt93/trident/commit/97f9945ae58ba1047fd770a7b36e1e5b5e3bff37))


### Bug Fixes

* **CI:** dependencies ([65e5df3](https://github.com/fdschmidt93/trident/commit/65e5df3fb85826c7842c8c78e4a812f226a48fc9))
* datamodule ([2ea6ced](https://github.com/fdschmidt93/trident/commit/2ea6ced0345f924ebba7e30738d952a78237599b))
* **datamodule:** default trident configuration ([60c62db](https://github.com/fdschmidt93/trident/commit/60c62db5217d4d203a5a16f10c4a7ec544480a88))
* docs ([7b0bc77](https://github.com/fdschmidt93/trident/commit/7b0bc77133b9fef3deb82177062ce42af8b1cf4c))
* **evaluation:** correctly pass dataset_name to prepare functions ([81d1023](https://github.com/fdschmidt93/trident/commit/81d10230f79a950bbdc2ec02951908e8f3e84fd2))
* **evaluation:** typo in check for step_collection_dico with datasets ([#7](https://github.com/fdschmidt93/trident/issues/7)) ([be9b33e](https://github.com/fdschmidt93/trident/commit/be9b33edac58f7a5006e1a0d18c7b9de3f5ba0cf))
* gradient accumulation taken care of by lightning ([3b46f39](https://github.com/fdschmidt93/trident/commit/3b46f399493cbfbeadbaaf2c1a74cf20dee15397))
* **hydra:** handling of non-primitives in _method_ ([2e8c7dc](https://github.com/fdschmidt93/trident/commit/2e8c7dc215787317964fa1c718f5e35f73463262))
* **hydra:** resolve parent-level links in config expansion ([498a75c](https://github.com/fdschmidt93/trident/commit/498a75c611435f520838fc4d56f0d4c824c77d50))
* include readme in index ([0703a17](https://github.com/fdschmidt93/trident/commit/0703a17ebb3602480c0118daf2059513e96ea1d1))
* **misc:** fixes from prior update ([823db71](https://github.com/fdschmidt93/trident/commit/823db7126111d84848a24e2edfde845d82a05751))
* missing default argument for module_cfg ([eeee5a3](https://github.com/fdschmidt93/trident/commit/eeee5a38bd41852102b8dd23c55fac33d54ac324))
* multiple eval dataloaders ([5afaf70](https://github.com/fdschmidt93/trident/commit/5afaf702a33a825ce5c56f511514f03efe2949d1))
* only trigger docgen for push to main ([a8c4e35](https://github.com/fdschmidt93/trident/commit/a8c4e35d54d863ad6873cba6ba9ac49c17650458))
* **optimizer:** assert type of trainer.max_epochs ([7c75de9](https://github.com/fdschmidt93/trident/commit/7c75de9765bf15eaf05feb90405364d65d7b69f4))
* **optimizer:** correct order of imports ([c2fd4f9](https://github.com/fdschmidt93/trident/commit/c2fd4f9f9465609e45adc90100ab0a0298fa735c))
* **run:** remove test_after_training ([a45c003](https://github.com/fdschmidt93/trident/commit/a45c00386ae9b1f3e649c7448a58e5609f964b3b))
* setup ([a91a10d](https://github.com/fdschmidt93/trident/commit/a91a10da11b2ae1a1880de760d2625e1e25e4661))
* setup ([a18c24b](https://github.com/fdschmidt93/trident/commit/a18c24b31f2e72feb54a62883c24eb11700b2c1a))
* styling of docs on gh-pages ([ede6d71](https://github.com/fdschmidt93/trident/commit/ede6d71489d8d74a7b6741f7710ac490e889f03b))
* styling of docs on gh-pages ([7f0a9c6](https://github.com/fdschmidt93/trident/commit/7f0a9c6325b6092326596077f5d1502507f3fb91))
* **train:** safeguard seed access ([668df26](https://github.com/fdschmidt93/trident/commit/668df2616ae6bc03318c4a983c30560db2af377d))
* **types:** annotation for dataloader getters ([6c7567f](https://github.com/fdschmidt93/trident/commit/6c7567f92fdfc7faef24899aa17ec4aa466126d9))


### Miscellaneous Chores

* **dataset:** merge get and get_raw to get ([d94ea7b](https://github.com/fdschmidt93/trident/commit/d94ea7b955a3c530a4c534ababecaf963aac0494))
