import time
import random
import string
import numpy as np
from flexi_hash_embedding import FlexiHashEmbedding


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
    assert np.allclose(c0, c1, atol=1e-6)
    # Assert that the same features in different rows are equal
    assert np.allclose(y[2], y[5])
    assert np.allclose(y[3], y[6])


def test_zalgo_text():
    """ Test weird input values: string, unicode, etc"""
    cases = ['𝔰𝔡𝔣𝔞', '𝖘𝖉𝖋𝖆', '👌👮  𝔰ᵈ𝔽𝔸  ✌😎', '𝓼𝓭𝓯𝓪', '𝓈𝒹𝒻𝒶', '𝕤𝕕𝕗𝕒',
            '•´¯`•» 𝐒∂𝔣𝐀➀ «•´¯`•', 's͓̽d͓̽f͓̽a͓̽1͓̽', 'ｓｄｆａ１　（雲ちも）']
    emb = FlexiHashEmbedding()
    X = [{case: 1} for case in cases]
    y = emb(X).data.cpu().numpy()
    for i, a in enumerate(y):
        for j, b in enumerate(y):
            if i == j:
                assert np.allclose(a, b)
            else:
                assert ~np.allclose(a, b)


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
    assert np.allclose(c1 * 2.0, c2, atol=1e-6)


def _gen_str(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))


def test_speed(n=64):
    X = [{_gen_str(): j for j in range(random.randint(0, 10))}
          for _ in range(4096)]
    emb = FlexiHashEmbedding()
    # Do a warmup forward-pass
    _ = emb(X).data.cpu().numpy()
    t0 = time.time()
    for _ in range(n):
        emb(X).data.cpu().numpy()
    t1 = time.time()
    assert (t1 - t0) / n < 100, "Average timing exceeded 100ms"
