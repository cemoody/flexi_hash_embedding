# Flexi Hash Embeddings

This PyTorch Module hashes variably-sized features
into a fixed-size embedding. This is useful in streaming
contexts for online learning.

If you have a variable-sized list of features, 
it can be extremely difficult to embed each feature
within pytorch. It can require poorly-supported sparse
matrix multiplication. 

So for example:
```
>>> X = [{'dog': 1, 'cat':2, 'elephant':4},
         {'dog': 2, 'run': 5}]
>>> embed = HashEmbedding()
>>> embed(X)
array([[ 0.,  0., -4., -1.,  0.,  2.],
       [ 0.,  0.,  0., -2., -5.,  0.]])

```

## Installation

Install locally by doing `git clone flexi`

## Testing

```
>>> pip install -e .
>>> py.test
```
