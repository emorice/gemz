"""
Linear model with a linear shrinkage
"""

import numpy as np

from gemz import linalg
from .linear import predict_loo as linear_predict_loo

def fit(data, prior_var):
    """
    Computes a representation of a linearily regularized precision matrix
    """

    _, len2 = data.shape

    regularized_covariance = linalg.RWSS(prior_var, data, 1./len2)

    return {
        'precision': np.linalg.inv(regularized_covariance)
        }

def predict_loo(model, new_data):
    """
    Prediction on new data, see `linear.predict_loo`
    """
    return linear_predict_loo(model=model, new_data=new_data)
