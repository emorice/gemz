"""
Data preprocessing utils
"""

import numpy as np

import gemz

def naive_rank_min(values, mask, val):
    """
    Number of masked values strictly inferior to val
    """
    return sum((values < val) * mask)

def naive_rank_max(values, mask, val):
    """
    Number of masked values inferior or equal to val
    """
    return sum((values <= val) * mask)

def naive_rank_avg(values, mask, val):
    """
    Average of rank_min and rank_mask
    """
    return 0.5 * (
            naive_rank_min(values, mask, val)
            +
            naive_rank_max(values, mask, val)
            )

def test_rank():
    """
    Ranking with awareness of ties and masks
    """

    max_value = 100
    num_values = 200

    rng = np.random.default_rng(0)
    values = rng.choice(max_value, size=num_values)
    mask = rng.choice(2, size=num_values)
    mask2 = rng.choice(2, size=num_values)


    naive_avg = [
        naive_rank_avg(values, mask, val)
        for val in values
        ]
    naive_avg2 = [
        naive_rank_avg(values, mask2, val)
        for val in values
        ]

    avg, avg2 = gemz.utils.average_rank(values, (mask, mask2))

    assert np.allclose(naive_avg, avg)
    assert np.allclose(naive_avg2, avg2)

def test_rank_multi():
    """
    Vectorized ranking of several arrays at once
    """

    max_value = 10
    num_values = 20

    rng = np.random.default_rng(0)
    values1 = rng.choice(max_value, size=num_values)
    values2 = rng.choice(max_value, size=num_values)
    mask = rng.choice(2, size=num_values)

    rank1 = gemz.utils.average_rank(values1, [mask])[0]
    rank2 = gemz.utils.average_rank(values2, [mask])[0]

    # Stack along first axis, rank along last (default)
    rank_last, = gemz.utils.average_rank( np.stack((values1, values2)),
            [mask[None, :]])
    assert np.allclose(rank_last, np.stack((rank1, rank2)))

    # Stack along last axis, rank along first (less common)
    rank_first, = gemz.utils.average_rank( np.stack((values1, values2), axis=-1),
            [mask[:, None]], axis=0)
    assert np.allclose(rank_first, np.stack((rank1, rank2), axis=-1))

def test_quantile_normalize():
    """
    Quantile normalization
    """
    qns, = gemz.utils.quantile_normalize(
            np.array([1, 2, 3]),
            np.array([[1, 1, 1]])
            )

    assert qns.shape == (3,)
    assert -10 < qns[0] < -0.1
    assert np.isclose(qns[1], 0.)
    assert 10 > qns[2] > 0.1

def test_masks():
    """
    CV masks
    """

    masks = gemz.utils.cv_masks(5, 10) # 5-fold, 10 samples

    assert masks.shape == (5, 10)
    assert np.all(np.sum(masks, 0) == 4) # Each sample present in all folds but one
    # Each fold contains four-fifths of the data
    assert np.all(np.sum(masks, 1) == 8)
