"""
Shrinkage models based on Ledoit--Wolf spectral deconvolution
"""

import numpy as np

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

import gemz.models.linear
from gemz import linalg

numpy2ri.activate()

def spectrum(data):
    """
    Spectrum obtained by numerically inverting the discretized spectral
    dispersion equations.

    Args:
        data: N1 x N2
    Returns:
        N1-long estimated spectrum
    """

    nlshrink = importr("nlshrink")

    # Working dimension last
    return np.array(nlshrink.tau_estimate(data.T))[::-1]

def fit(data):
    """
    Estimates spectrum and builds a representation of the corresponding
    precision matrix
    """
    _, len2 = np.shape(data)
    left, _, _ = np.linalg.svd(data)

    opt_spectrum = spectrum(data)

    return {
        'precision': linalg.RWSS(0., left, np.diag(len2 / opt_spectrum)),
        'spectrum': opt_spectrum
        }

def predict_loo(model, new_data):
    """
    See linea.predict_loo
    """
    return gemz.models.linear.predict_loo(model, new_data)
