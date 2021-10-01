# TODO
## General
- [ ] Documentation: docgen?
- [ ] Possibly convert Huggingface datasets to Pandas for less noisy and faster preprocessing
- [ ] Structured configs, better config validation
- [ ] partial overrides?

## Model
- [ ] Mixin on-the-fly?
      
## Datasets
- [X] Make dataset across tasks arbitrarily combinable (train, val, test)
- [ ] Keep raw dataset? Check mem requirements
- [ ] Better semantics for setup

## Evaluation
- [ ] Support non-classification evaluation loops

## Tatoeba
- Need
    - [ ] Embed/Forward & Collator differentiate between source and target language
    - [ ] Collect embeddings
    - [ ] Collect indices
    - [ ] Profit?
