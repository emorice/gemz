"""
Linear model with a linear shrinkage
"""

import numpy as np

from gemz import linalg
from .linear import (
    predict_loo as linear_predict_loo,
    spectrum as linear_spectrum
    )

def fit(data, prior_var, target=None, scale=None):
    """
    Computes a representation of a linearily regularized precision matrix
    """

    len1, len2 = data.shape

    if target is None:
        target = 1.

    if scale is None:
        scale = 1.

    scale = scale + np.zeros(len1)

    regularized_covariance = linalg.RWSS(prior_var * target, scale[:, None] * data, 1./len2)

    return {
        'precision': np.linalg.inv(regularized_covariance),
        'prior_var': prior_var
        }

def spectrum(data, prior_var):
    """
    Spectrum implicitely used by the regularized model when target and scale are None
    """
    orig_spectrum = linear_spectrum(data)
    adjusted_spectrum = orig_spectrum + prior_var
    return adjusted_spectrum

def predict_loo(model, new_data):
    """
    Prediction on new data, see `linear.predict_loo`
    """
    return linear_predict_loo(model=model, new_data=new_data)
