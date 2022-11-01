"""
Information-driven Gaussian mixture models
"""

import numpy as np
import distrax
import jax.numpy as jnp

from gemz import jax_utils
from gemz.jax_numpy import jaxify
from . import methods, gmm, cv

methods.add_module('igmm', __name__)

# Public interface
# ================

def fit(data, n_groups, seed=0, barrier_strength=1e-2, init_resps=None):
    """
    Learns a GMM with an information loss

    Args:
        data: len1 x len2, where the len2 cols will be clustered.
    """

    if init_resps is None:
        # Random uniform initialization
        rng = np.random.default_rng(seed)
        resp0 = rng.uniform(size=(n_groups, data.shape[-1]))
        resp0 /= resp0.sum(axis=0)
    else:
        resp0 = init_resps

    max_results = jax_utils.maximize(
        _igmm_obj,
        init={
            'responsibilities': resp0,
            'reg_covar': 1.,
            },
        data={
            'data': data,
            'barrier_strength': barrier_strength,
            },
        bijectors={
            'responsibilities': jax_utils.Softmax(),
            'reg_covar': distrax.Lambda(jnp.exp),
            },
        scipy_method='L-BFGS-B',
        )

    resps = max_results['opt']['responsibilities']
    groups = np.argmax(
        resps,
        axis=0
        )

    return gmm.precompute_loo({
        'opt': max_results,
        'data': data,
        'groups': groups,
        'responsibilities': resps,
        'reg_covar': max_results['opt']['reg_covar']
        })

predict_loo = gmm.predict_loo

cv = cv.Int1dCV('n_groups', 'components')

get_name = gmm.get_name

# Objective functions
# ===================

@jaxify
def _igmm_obj(data, responsibilities, reg_covar, barrier_strength=0.1):
    """
    Information-GMM objective

    Args:
        data: N x P, P being the large dimension
        responsibilities: K x P, K being the number of groups
    """

    precomp = gmm.precompute_loo(dict(
        data=data,
        reg_covar=reg_covar,
        responsibilities=responsibilities
        ))

    data_np = data
    responsibilities_kp = responsibilities
    group_sizes_k = np.sum(responsibilities_kp, -1)
    len1 = data.shape[0]

    means_kpn = precomp['means']
    covariances_kpnn = precomp['covariances']
    precisions_kpnn = precomp['precisions']

    centered_data_kpn = data_np.T - means_kpn

    # sum over N
    misfits_kp = np.sum(
        (precisions_kpnn @ centered_data_kpn[..., None])
        * centered_data_kpn[..., None],
        (-2, -1)
        )

    # batched det over N x N
    _signs, log_det_covs_kp = np.linalg.slogdet(covariances_kpnn)

    # scalar
    #agg_misfits = np.tensordot(responsibilities, misfits)

    avg_misfits_k = (
            np.sum(misfits_kp * responsibilities_kp, -1)
            / (len1 * group_sizes_k)
            )

    # scalar
    exp_log_lk = (
        - 0.5 * np.tensordot(responsibilities_kp, log_det_covs_kp)
        - 0.5 * np.sum(len1 * group_sizes_k * np.log(avg_misfits_k))
        )
    # scalar
    entropy = - np.tensordot(responsibilities, np.log(responsibilities))

    # scalar
    barrier = barrier_strength * np.sum(np.log(responsibilities))

    return exp_log_lk + entropy + barrier

@jaxify
def _gmm_obj(data, responsibilities, barrier_strength=0.1):
    """
    Classical GMM objective for reference.

    Args:
        data: N x P, P being the large dimension
        responsibilities: K x P, K being the number of groups
    """
    precomp = gmm.precompute(dict(data=data, responsibilites=responsibilities))

    group_sizes = precomp['group_sizes'] # k
    means = precomp['means'] # kn
    precisions = precomp['precisions'] # knn

    # This requires O(KNP) storage, should we do better ?
    # K x N x P
    centered_data = data - means[:, :, None]
    # K x P, sum over N
    misfits = np.sum(
        (precisions @ centered_data)
        * centered_data,
        1
        )

    # K, det over N x N
    _signs, log_det_precs = np.linalg.slogdet(precisions)

    agg_misfits = np.tensordot(responsibilities, misfits)
    # scalar
    exp_log_lk = (
        + 0.5 * group_sizes @ log_det_precs
        - 0.5 * agg_misfits
        )
    # scalar
    entropy = - np.tensordot(responsibilities, np.log(responsibilities))

    # scalar
    barrier = barrier_strength * np.sum(np.log(responsibilities))

    return exp_log_lk + entropy + barrier
