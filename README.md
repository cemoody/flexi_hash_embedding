# Flexi Hash Embeddings

This PyTorch Module hashes and sums variably-sized dictionaries of 
features into a single fixed-size embedding. 
Feature keys are hashed, which
is ideal for streaming contexts and online-learning
such that we don't have to memorize a
mapping between feature keys and indices.


So for example:

```python
>>> X = [{'dog': 1, 'cat':2, 'elephant':4},
         {'dog': 2, 'run': 5}]
>>> from flexi_hash_embedding import FlexiHashEmbedding
>>> embed = FlexiHashEmbedding(dim=5)
>>> embed(X)
tensor([[ 2.5842e+00,  1.9553e+01,  1.0246e+00,  2.2797e+01,  1.7812e+01],
        [-6.2967e+00,  1.4947e+01, -2.6539e+01, -1.4348e+01, -6.7396e-01]])
```

![img](https://i.imgur.com/OBkiM7T.png)

## Speed

A large batchsize of 4096 with on average 5 features per row equates
to about 20,000 total features. This module will hash that many features
in about 20ms on a modern MacBook Pro.

## Installation

Install from PyPi do `pip install flexi_hash_embedding`

Install locally by doing `git@github.com:cemoody/flexi_hash_embedding.git`.

## Testing

```
>>> pip install -e .
>>> py.test
```
