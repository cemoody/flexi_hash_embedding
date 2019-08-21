# Flexi Hash Embeddings

This PyTorch Module hashes and sums variably-sized dictionaries of 
features into a single fixed-size embedding. 
Feature keys are hashed, which
is ideal for streaming contexts and online-learning
such that we don't have to memorize a
mapping between feature keys and indices.
Multiple variable-length features are grouped by example
and then summed.

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

## Example
Frequently, we'll have data for a single event or user
that's, for example, in a JSON blob format.
Features may be missing, incomplete or never before seen.
Furthermore, there's a variable number of features defined.

So even if we have a total of six features spread

![img](https://i.imgur.com/OBkiM7T.png)

In the example above we have a total of six features but they're
spread out across three clients. The first client has three active
features, the second client two features 
(and only one feature that overlaps with the first client)
and the third client has a single feature active.
`Flexi Hash Embeddings` returns three vectors, one for each client,
and not six vectors even though there are six features present.
The first client's vector is a sum of three feature vectors 
(plus_flag, age_under, luxe_flag) 
while the second client's vector is a sum of just two feature vectors
(lppish, luxe_flag)
and the third client's vector is just a single feature.

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

To publish a new package version:

```bash
python3 setup.py sdist bdist_wheel
twine upload dist/*
pip install --index-url https://pypi.org/simple/ --no-deps flexi_hash_embedding
```


