"""
Regularized linear model obtained by keeping only some factors of a singular
value decompistion
"""

import logging
import numpy as np

from gemz import linalg
from .methods import add_module
from . import linear, svd

add_module('peer', __name__)

def fit(data, n_factors):
    """
    Builds a representation of the precision matrix by using `n_factors`
    inferred PEER factors
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

# All other methods are inherited from the SVD case
predict_loo = svd.predict_loo
get_name = svd.get_name
make_grid = svd.make_grid
get_grid_axis = svd.get_grid_axis
