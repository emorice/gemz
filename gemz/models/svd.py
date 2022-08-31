"""
Regularized linear model obtained by keeping only some factors of a singular
value decompistion
"""

import logging
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

def make_grid(partial_spec, data, grid_size=None):
    """
    Simple 1-2-5 grid, size depends only on data shape.
    """

    if grid_size is not None:
        logging.warning('Ignored argument grid_size')

    max_size = min(data.shape)

    sizes = []
    base = 1
    while True:
        for fact in (1, 2, 5):
            size = base * fact
            if size >= max_size:
                break
            sizes.append(size)
        if size >= max_size:
            break
        base *= 10

    return [
        dict(partial_spec,
            n_factors=size)
        for size in sizes
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
