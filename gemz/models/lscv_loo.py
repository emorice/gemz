"""
Linear shrinkage model optimized through a leave-one-out objective function
"""

import numpy as np

from gemz import linalg
from gemz.jax_numpy import jaxify
from gemz.jax_utils import minimize, RegExp

from . import linear
from .methods import add_module

add_module('lscv_loo', __name__)

# Public interface
# ================

def fit(data, loss='rss', prior_var_min=0.):
    """
    Fit a linear shrinkage model by loo cv ad.

    Args:
        data: N1 x N2, expecting N1 < N2. Loo is done over N1.
        loss: one of the following internal losses:
            'rss': L2 distance, sum of squares of residuals, aka rss.
            'indep': product of likelihoods per coordinate, using the
                predicted variance from the model.
            'joint': joint log-likelihood from the model
        prior_var_min: lower bound on the regularization to use. Very small
            regularization can cause fatal numerical failure of the loo
            computation and subsequent optimization.
    """

    _loss = {
        'rss': _loo_loss_rss,
        'indep': _loo_loss_indep,
        'joint': _loo_loss_joint,
        }[loss]

    opt = minimize(
            jaxify(_loss),
            init={
                'prior_var': 1.0,
                },
            data={
                'data': data,
                },
            bijectors={
                'prior_var': RegExp(prior_var_min)
                },
            #jit=False,
            )

    regularized_covariance = linalg.SymmetricLowRankUpdate(
        opt['opt']['prior_var'],
        data.T,
        1./data.shape[0]
        )

    return {
            # Warning: precision here is off by a global factor
            'precision': np.linalg.inv(regularized_covariance),
            'opt': opt
            }

predict_loo = linear.predict_loo

def get_name(spec):
    """
    Short name
    """
    return f'{spec["model"]}/{spec["loss"] if "loss" in spec else "rss"}'

# Internals
# =========

def _loo_local_precisions(data_np, prior_var):
    """
    Virtual stack of leave-one-out precision estimator, up to a global scale
    """

    base_gram_pp = linalg.SymmetricLowRankUpdate(
            prior_var, data_np.T, 1. / (data_np.shape[0] - 1)
            )

    loo_ugram_npp = linalg.SymmetricLowRankUpdate(
            base_gram_pp,
            # n is now a batching dimension, and we add a dummy contracting
            # dimension at the end
            data_np[..., None],
            # Downdate
            -1. / (data_np.shape[0] - 1))

    loo_uprec_npp = np.linalg.inv(loo_ugram_npp)

    return loo_uprec_npp

def _loo_predict(data, prior_var):
    """
    Computes leave-one-out predictions
    """

    # Naming scheme: data shape is (n, p)
    data_np = data

    loo_prec_npp = _loo_local_precisions(data_np, prior_var)

    # n-batched pp-matrix p-vector
    local_residuals_np = (loo_prec_npp @ data_np[..., None])[..., 0]

    # n-batched diagonal
    local_scales_np = np.diagonal(loo_prec_npp)

    residuals = local_residuals_np / local_scales_np

    precisions = local_scales_np
    # At this point, `precisions` is still off by a global factor. The optimal
    # scale technically depends on the loss function, but we already can set a
    # reasonable default
    precisions /= np.mean(residuals**2 * precisions)

    return residuals, precisions

def _loo_loss_rss(data, prior_var):
    """
    Get leave-one-out predictions and apply a loss function
    """
    residuals, _precisions = _loo_predict(data, prior_var)
    return np.sum(residuals**2)

def _loo_loss_indep(data, prior_var):
    residuals, precisions = _loo_predict(data, prior_var)
    return np.sum((residuals**2) * precisions - np.log(precisions))

def _loo_loss_joint(data, prior_var):
    data_np = data

    local_prec_npp = _loo_local_precisions(data_np, prior_var)

    misfits_n = (data_np[..., None, :] @ (local_prec_npp @ data_np[..., None]))[..., 0, 0]

    global_scale = np.mean(misfits_n)
    _, log_dets_n = np.linalg.slogdet(local_prec_npp)

    losses_n = data_np.shape[-1] * np.log(global_scale) - log_dets_n

    return np.sum(losses_n)
