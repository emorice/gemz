"""
Data pre-processing utils
"""

import numpy as np
from scipy.special import ndtri

def average_rank(values, masks, axis=-1):
    """
    Average masked ranked of each value, for each given mask, along the given
    axis of values.

    The average masked rank of a value is the average of the number of masked values
    less than the given value, and the number of masked values less than or equal
    to the given value.

    The argument `masks` is simply iterated over with no vectorization benefits,
    but sorting the values is only done once for all masks, hence the interest
    of computing several masks at once.

    To get the empirical unregularized CDF, simply divide the resulting ranks by
    the number of masked values.
    """
    sorter = np.argsort(values, axis=axis)

    ranks = [ 0. for _ in masks]
    for flip in False, True:
        if flip:
            _sorter = np.flip(sorter, axis=axis)
        else:
            _sorter = sorter
        sorted_values = np.take_along_axis(values, _sorter, axis=axis)


        if not flip:
            is_new = sorted_values > np.roll(sorted_values, 1, axis=axis)
        else:
            is_new = sorted_values < np.roll(sorted_values, 1, axis=axis)

        for i, mask in enumerate(masks):
            sorted_mask = np.take_along_axis(mask, _sorter, axis=axis)
            sorted_rank_min = np.maximum.accumulate(
                    is_new * (
                        np.cumsum(sorted_mask, axis=axis) - sorted_mask
                        ),
                    axis=axis)
            rank_min = np.empty_like(values)
            np.put_along_axis(rank_min, _sorter, sorted_rank_min, axis=axis)
            if flip:
                rank_min = np.sum(mask, axis=axis) - rank_min
            ranks[i] += 0.5 * rank_min

    return ranks

def quantile_normalize(values, masks, axis=-1, pseudocounts=0.5):
    """
    Quantile normalize the values with quantiles computed only with respect to
    the masked values.

    The empirical CDF is regularized by assuming extra `pseudocounts` samples at
    plus and minus infinity, so that values outside of the training support are
    still mapped to (somewhat arbitrary) finite values.
    """
    ranks = average_rank(values, masks, axis=axis)

    return [
        ndtri(
            (rank + pseudocounts) / (np.sum(mask, axis=axis) + 2 * pseudocounts)
            )
        for mask, rank in zip(masks, ranks)
        ]

def cv_masks(fold_count, sample_count, seed=0):
    """
    Generate random masks.
    """
    rng = np.random.default_rng(seed)

    random_rank = rng.choice(sample_count, sample_count, replace=False)

    fold_indexes = np.arange(fold_count)

    masks = random_rank % fold_count != fold_indexes[:, None]

    return masks
