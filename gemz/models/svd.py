"""
Regularized linear model obtained by keeping only some factors of a singular
value decompistion
"""

import numpy as np

from gemz import linalg
from .methods import add_module
from . import linear, cv

add_module('svd', __name__)

def get_spec(n_factors: int | None = None, revision: int = 2):
    """
    Get spec with latest recommended defaults
    """
    spec = {'model': 'svd', 'revision': revision}
    if n_factors is not None:
        spec['n_factors'] = n_factors
    return spec

def fit(data, n_factors, revision=1):
    """
    Builds a representation of the precision matrix by using only the first
    `n_factors` of the SVD of data.
    """

    len1, len2 = data.shape

    _left, spectrum, right_t = np.linalg.svd(data, full_matrices=False)

    if revision == 1:
        # Legacy code, contains errors about spectrum calculations
        residual_variance = np.sum(spectrum[n_factors:]**2) / data.size
        cov = linalg.SymmetricLowRankUpdate(
                residual_variance,
                right_t[:n_factors].T,
                spectrum[:n_factors]**2 / len1
                )
    elif revision == 2:
        # First fix: we only average over the n - k last eigenvalues
        residual_variance = np.sum(spectrum[n_factors:]**2) / (
                len1 * (len2 - n_factors)
                )
        # Second fix: the base matrix has all dimensions, so the update need to
        # only add the rest of the matched eigenvalues
        cov = linalg.SymmetricLowRankUpdate(
                residual_variance,
                right_t[:n_factors].T,
                spectrum[:n_factors]**2 / len1 - residual_variance
                )
    else:
        raise ValueError('revision', revision)
    return {
        'precision': np.linalg.inv(cov)
        }

predict_loo = linear.predict_loo

def get_name(spec):
    """
    Descriptive name
    """
    rev = spec.get('revision', 1)
    name = f"{spec['model']}{rev if rev != 1 else ''}"
    if 'n_factors' in spec:
        return f"{name}/{spec['n_factors']}"
    return name

cv = cv.Int1dCV('n_factors', 'factors')
