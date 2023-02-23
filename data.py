"""
Collection of test dataset generators
"""

from dataclasses import dataclass

from collections.abc import Collection

import numpy as np
import scipy.stats
import pandas as pd

def gen_2d_gaussian(size, rng, **params):
    """
    Random 2d gaussian
    """

    data_x = rng.normal(params['mean_x'], params['std_x'], size=size)
    data_y = (
            data_x
            + params['intercept']
            + rng.normal(0., params['std_error'], size=size)
            )
    return data_x, data_y


def rectangle(low, high):
    """
    Multivariate rectangular uniform parametrized by its bounds.
    """
    return _Rectangle(low, high)


class _Rectangle:
    def __init__(self, low, high):
        locs = np.asarray(low)
        scales = np.asarray(high) - locs
        self.dim = len(scales)
        self._dist = scipy.stats.uniform(locs, scales)

    def rvs(self, size, random_state=None):
        """
        Generate random variates
        """
        _size = (size, self.dim)
        return self._dist.rvs(_size, random_state=random_state)

def mixture(distributions, weights):
    """
    Mixture of several base distributions
    """
    return _Mixture(distributions, weights)

@dataclass
class _Mixture:
    distributions: Collection
    weights: Collection

    def rvs(self, size, random_state):
        """
        Naive generator
        """
        rng = np.random.default_rng(random_state)
        categories = rng.choice(
                len(self.distributions),
                size=size
                )

        candidates = np.stack([
                dist.rvs(size=size, random_state=rng)
                for dist in self.distributions
                ])

        samples, = np.take_along_axis(candidates, categories[None, :, None], axis=0)

        return categories, samples

def mixture_to_df(labels, samples, names, dim_names):
    """
    Pack random samples generated from a mixture into a data frame
    """
    return (pd
        .DataFrame(samples.T, columns=dim_names)
        .assign(num_label=labels)
        .merge(pd.Series(names, name='label'), left_on='num_label',
            right_index=True)
        .sort_index()
        .drop('num_label', axis=1)
        )
