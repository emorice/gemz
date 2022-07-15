"""
Diagonal-target linear shrinkage with the whole diagonal used as a CV parameter
"""

import numpy as np

from gemz.jax_utils import minimize
from gemz.jax_numpy import jaxify
from gemz.models import cv, linear_shrinkage

def fit(data, scale=1.):
    """
    CV opt of diagonal shrinkage target

    Args:
        data: N1 x N2. A N2 x N2 implicit covariance is sought with a variable
            N2-long diagonal regularization.
    """

    init = dict(
        log_diagonal=np.zeros(data.shape[1]),
        )
    opt_data = dict(
        data=data
        )

    if scale is None:
        init['log_scale'] = np.zeros(data.shape[1])
    else:
        opt_data['log_scale'] = np.log(scale)

    opt = minimize(
        cv_loss,
        init=init, data=opt_data,
        scipy_method='L-BFGS-B',
        )

    final_log_scale = (
        (opt['opt'] if scale is None else opt_data)
        ['log_scale']
        )

    model = linear_shrinkage.fit(
        data,
        prior_var=1.,
        target=np.exp(opt['opt']['log_diagonal']),
        scale=np.exp(final_log_scale)
        )

    return {
        'model': model,
        'opt': opt
        }

@jaxify
def cv_loss(log_diagonal, log_scale, data):
    """
    Cross-validated geometric loss
    """
    # Note that prior_var is set to 1 since we already have all the freedom
    # we need in the diagonal itself
    return cv.fit_cv(
        data, linear_shrinkage,
        fold_count=10, loss_name="GEOM",
        target=np.exp(log_diagonal),
        scale=np.exp(log_scale),
        prior_var=1.
        )

def predict_loo(model, new_data):
    """
    See linear_shrinkage
    """
    return linear_shrinkage.predict_loo(model['model'], new_data)
