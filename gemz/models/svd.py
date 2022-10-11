"""
Regularized linear model obtained by keeping only some factors of a singular
value decompistion
"""

import numpy as np

from gemz import linalg
from .methods import add_module
from . import linear

add_module('svd', __name__)

def fit(data, n_factors):
    """
    Builds a representation of the precision matrix by using only the first
    `n_factors` of the SVD of data.
    """

    len1, _len2 = data.shape

    _left, spectrum, right_t = np.linalg.svd(data, full_matrices=False)

    residual_variance = np.sum(spectrum[n_factors:]**2) / data.size

    cov = linalg.SymmetricLowRankUpdate(
            residual_variance,
            right_t[:n_factors].T,
            spectrum[:n_factors]**2 / len1
            )

    return {
        'precision': np.linalg.inv(cov)
        }

predict_loo = linear.predict_loo

def get_name(spec):
    """
    Descriptive name
    """
    return f"{spec['model']}/{spec['n_factors']}"

def make_grid(partial_spec, data, grid_size):
    """
    Simple logarithmic scale of not more than grid_size entry.

    Grid can be smaller than requested.
    """

    max_size = min(data.shape)

    sizes = np.unique(
            np.int32(np.floor(
                np.exp(
                    np.linspace(0., np.log(max_size), grid_size)
                    )
                ))
            )

    return sizes

def make_grid_specs(partial_spec, grid):
    """
    Generate model specs from grid values
    """
    return [
        dict(partial_spec,
            n_factors=int(size))
        for size in grid
        ]

def get_grid_axis(specs):
    """
    Compact summary of the variable parameter of a list of models
    """
    return {
        'name': 'factors',
        'log': True,
        'values': [ s['n_factors'] for s in specs ]
        }
