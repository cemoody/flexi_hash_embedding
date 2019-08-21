import numpy as np
from flexi_hash_embedding.flexi_hash_embedding import FlexiHashEmbedding


def test_sum():
    """ Test summing of different features
    """
    emb = FlexiHashEmbedding()
    X = [{'age': 16, 'spend': -2, 'height':4},
         {'age': 2, 'boho': 1},
         {'height': 4},
         {'boho': 1},
         {'age': 16, 'spend': -2, 'boho':1},
         {'height': 4},
         {'boho': 1}]
    y = emb(X).data.cpu().numpy()
    # c0 = age + spend + height - height
    c0 = y[0] - y[2]
    # c1 = age + spend + boho - boho
    c1 = y[4] - y[3]
    assert np.allclose(c0, c1)
    # c2 = age + spend
    c2 = y[1]
    assert np.allclose(c0, c2)
    # Assert that the same features in different rows are equal
    c5, c6 = y[5], y[6]
    assert c5 == c6


def test_zalgo_text():
    """ Test weird input values: string, unicode, etc"""
    cases = ['𝔰𝔡𝔣𝔞', '𝖘𝖉𝖋𝖆', '👌👮  𝔰ᵈ𝔽𝔸  ✌😎', '𝓼𝓭𝓯𝓪', '𝓈𝒹𝒻𝒶', '𝕤𝕕𝕗𝕒',
            '•´¯`•» 𝐒∂𝔣𝐀➀ «•´¯`•', 's͓̽d͓̽f͓̽a͓̽1͓̽', 'ｓｄｆａ１　（雲ちも）']
    emb = FlexiHashEmbedding()
    for case in cases:
        X = [{case: 1}, {case: 2}]
        y = emb(X).data.cpu().numpy()
        assert np.allclose(y[0], y[1])


def test_zero():
    """ Test linearity of values
    """
    emb = FlexiHashEmbedding()
    X = [{'boho': 0},
         {'boho': 1},
         {'boho': 2}]
    y = emb(X).data.cpu().numpy()
    c0 = y[0]
    c1 = y[1] 
    c2 = y[2]
    assert np.allclose(c0, c1 * 0.0)
    assert np.allclose(c1 * 2.0, c2)
