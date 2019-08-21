# Flexi Hash Embeddings

This PyTorch Module hashes variably-sized dictionaries of 
features into a single fixed-size embedding. 
This is particularly useful in streaming
contexts for online learning.


So for example:

```
>>> X = [{'dog': 1, 'cat':2, 'elephant':4},
         {'dog': 2, 'run': 5}]
>>> from flexi_hash_embedding import FlexiHashEmbedding
>>> embed = FlexiHashEmbedding(dim=5)
>>> embed(X)
tensor([[ 2.5842e+00,  1.9553e+01,  1.0246e+00,  2.2797e+01,  1.7812e+01],
        [-6.2967e+00,  1.4947e+01, -2.6539e+01, -1.4348e+01, -6.7396e-01]])
```

!(diagram)[diagram.png]

## Speed

A large batchsize of 4096 with on average 5 features per row equates
to about 20,000 features, and this module will hash that many features
in about 20ms on a modern MacBook Pro.

## Installation

Install from PyPi

Install locally by doing `git@github.com:cemoody/flexi_hash_embedding.git`.

## Testing

```
>>> pip install -e .
>>> py.test
```
