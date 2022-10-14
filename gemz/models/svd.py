"""
Regularized linear model obtained by keeping only some factors of a singular
value decompistion
"""

import numpy as np

from gemz import linalg
from .methods import add_module
from . import linear, cv

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

cv = cv.Int1dCV('n_factors', 'factors')
