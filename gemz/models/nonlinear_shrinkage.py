"""
Shrinkage models based on Ledoit--Wolf spectral deconvolution
"""

import numpy as np

from gemz import linalg
from . import methods, linear

methods.add_module('nonlinear_shrinkage', __name__)

def spectrum(data):
    """
    Spectrum obtained by numerically inverting the discretized spectral
    dispersion equations.

    Args:
        data: N1 x N2
    Returns:
        N2-long estimated spectrum
    """

    # pylint: disable=import-outside-toplevel
    # Deliberate, rpy2 starts a R session in the background,
    # which we want to skip if this model is never used
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri, default_converter
    from rpy2.robjects.conversion import localconverter

    nlshrink = importr("nlshrink")

    with localconverter(default_converter + numpy2ri.converter):
        # Working dimension last
        return np.array(nlshrink.tau_estimate(data))[::-1]

def fit(data):
    """
    Estimates spectrum and builds a representation of the corresponding
    precision matrix

    Args:
        data: N1 x N2, N1 < N2
    """
    len1, len2 = np.shape(data)
    _, _, right = np.linalg.svd(data, full_matrices=False)

    cov_spectrum = spectrum(data)
    inv_spectrum = 1.0 / cov_spectrum

    null_precision = inv_spectrum[-1]

    assert np.allclose(inv_spectrum[len1:len2], null_precision)

    return {
        'precision': linalg.SymmetricLowRankUpdate(
            null_precision,
            right.T,
            np.diag(inv_spectrum[:len1] - null_precision)
            ),
        'spectrum': cov_spectrum
        }

predict_loo = linear.predict_loo
