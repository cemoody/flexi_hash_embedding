# Flexi Hash Embeddings

This PyTorch Module hashes and sums variably-sized dictionaries of 
features into a single fixed-size embedding. 
Feature keys are hashed, which
is ideal for streaming contexts and online-learning
such that we don't have to memorize a
mapping between feature keys and indices.

Uses the wonderful [torch scatter](https://github.com/rusty1s/pytorch_scatter) library
and sklearn's feature hashing under the hood.

So for example:

```python
>>> from flexi_hash_embedding import FlexiHashEmbedding
>>> X = [{'dog': 1, 'cat':2, 'elephant':4},
         {'dog': 2, 'run': 5}]
>>> embed = FlexiHashEmbedding(dim=5)
>>> embed(X)
tensor([[  1.0753,  -5.3999,   2.6890,   2.4224,  -2.8352],
        [  2.9265,   5.1440,  -4.1737, -12.3070,  -8.2725]],
       grad_fn=<ScatterAddBackward>)
```

![img](https://i.imgur.com/OBkiM7T.png)

## Speed

A large batchsize of 4096 with on average 5 features per row equates
to about 20,000 total features. This module will hash that many features
in about 20ms on a modern MacBook Pro.

## Installation

Install from PyPi do `pip install flexi-hash-embedding`

Install locally by doing `git@github.com:cemoody/flexi_hash_embedding.git`.

## Testing

```bash
>>> pip install -e .
>>> py.test
```

To publish a new package:

```bash
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository-url https://pypi.org/legacy/ dist/*
pip install --index-url https://pypi.org/simple/ --no-deps flexi_hash_embedding
Make sure to specify your username in the package name!


