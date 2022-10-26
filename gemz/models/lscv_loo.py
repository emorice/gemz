"""
Linear shrinkage model optimized through a leave-one-out objective function
"""

import numpy as np
import jax.numpy as jnp
import distrax

from gemz import linalg
from gemz.jax_numpy import jaxify
from gemz.jax_utils import minimize

from . import linear
from .methods import add_module

add_module('lscv_loo', __name__)

# Public interface
# ================

def fit(data):
    """
    Fit a linear shrinkage model by loo cv ad.
    """

    opt = minimize(
            _loo_loss,
            init={
                'prior_var': 1.0,
                },
            data={
                'data': data
                },
            bijectors={
                'prior_var': distrax.Lambda(jnp.exp)
                }
            )

    regularized_covariance = linalg.SymmetricLowRankUpdate(
        opt['opt']['prior_var'],
        data.T,
        1./data.shape[0]
        )

    return {
            'precision': np.linalg.inv(regularized_covariance),
            'opt': opt
            }

predict_loo = linear.predict_loo

# Internals
# =========

def _loo_predict(data, prior_var):
    """
    Computes leave-one-out predictions
    """

    # Naming scheme: data shape is (n, p)
    data_np = data

    base_ugram_pp = linalg.SymmetricLowRankUpdate(
            prior_var * (data.shape[0] - 1), data_np.T, 1.
            )

    loo_ugram_npp = linalg.SymmetricLowRankUpdate(
            base_ugram_pp,
            # n is now a batching dimension, and we add a dummy contracting
            # dimension at the end
            data_np[..., None],
            # Downdate
            -1.)

    loo_uprec_npp = np.linalg.inv(loo_ugram_npp)

    # n-batched pp-matrix p-vector
    unscaled_residuals_np = (loo_uprec_npp @ data_np[..., None])[..., 0]

    # n-batched diagonal
    scales_np = np.diagonal(loo_uprec_npp)

    residuals = unscaled_residuals_np / scales_np
    variances = (data.shape[0] - 1) / scales_np

    return residuals, variances

@jaxify
def _loo_loss(data, prior_var):
    """
    Get leave-one-out predictions and apply a loss function

    Args:
        data: N1 x N2, expecting N1 < N2. Loo is done over N1.
    """
    residuals, _variances = _loo_predict(data, prior_var)

    return np.sum(residuals**2)
