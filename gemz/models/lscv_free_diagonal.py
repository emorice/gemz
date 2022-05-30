"""
Diagonal-target linear shrinkage with the whole diagonal used as a CV parameter
"""

import numpy as np

from gemz.jax_utils import minimize
from gemz.jax_numpy import indirect_jax
from gemz.models import cv, linear_shrinkage

def fit(data):
    """
    CV opt of diagonal shrinkage target
    """

    opt = minimize(
        cv_loss,
        init=dict(
            log_diagonal=np.zeros(data.shape[0])
            ),
        data=dict(
            data=data
            )
        )

    return linear_shrinkage.fit(
        data,
        prior_var=1.,
        target=np.exp(opt['opt']['log_diagonal'])
        )

@indirect_jax
def cv_loss(log_diagonal, data):
    """
    Cross-validated geometric loss
    """
    # Note that prior_var is set to 1 since we already have all the freedom
    # we need in the diagonal itself
    return cv.fit_cv(
        data, linear_shrinkage,
        fold_count=10, loss_name="GEOM",
        target=np.exp(log_diagonal),
        prior_var=1.
        )

def predict_loo(model, new_data):
    """
    See linear_shrinkage
    """
    return linear_shrinkage.predict_loo(model, new_data)
