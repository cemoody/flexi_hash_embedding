import torch
import numpy as np
from torch import nn
from scipy.sparse import coo_matrix
from torch_scatter import scatter_add
from sklearn.feature_extraction import FeatureHasher


class FlexiHashEmbedding(nn.Module):
    def __init__(self, n_features=1048576, dim=128):
        """Differentiably embed and sum a variable-length number of features
        a single fixed-size embedding per row. Feature keys are hashed, which
        is ideal for online-learning such that we don't have to memorize a
        mapping between feature keys and indices.
        Feature embeddings are then scaled linearly by their values.

        Arguments
        ---------
        n_hashes: int
        p: float < 0
            Pick n_features to minimize hash collisions to a probability p. 
            The number of
            unique hashes n=sqrt(2 * n_features * p)

        >>> X = [{'age': 16, 'spend': -2, 'height':4}, {'age': 2, 'boho': -5}]
        >>> embed = HashEmbedding()
        >>> embed(X)
        array([[ 0.,  0., -4., -1.,  0.,  2.],
               [ 0.,  0.,  0., -2., -5.,  0.]])
        """
        super().__init__()
        self.n_features = n_features
        self.hasher = FeatureHasher(self.n_features, alternate_sign=False)
        self.embed = nn.Embedding(n_features, dim)

    def _move(self, arr, device=None):
        """ Transfer the input numpy array to the correct device
        """
        if device is None:
            device = self.embed.weight.data.device
        tarr = torch.from_numpy(arr).to(device)
        return tarr

    def __call__(self, X):
        Xcsr = self.hasher.transform(X)
        Xcoo = Xcsr.tocoo()
        # feature_index is of length n_total_features, 
        # e.g. the size of the concatenated list
        # of all variable-length features.
        # It is not of shape X.shape[1].
        # The values are hashed to 0 <= val <= n_features
        feature_index = self._move(Xcoo.col.astype(np.int64))
        # embed_index is of size number of rows, e.g. n_rows=len(X)
        embed_index = self._move(Xcoo.row.astype(np.int64))
        # This is of size (n_total_features, n_dim)
        feature_embed = self.embed(feature_index)
        # To understand scatter add see:
        # https://pytorch-scatter.readthedocs.io/en/latest/functions/add.html
        # summed is of size (n_rows, n_dim)
        summed = scatter_add(feature_embed.t(), embed_index[None, :]).t()
        return summed
