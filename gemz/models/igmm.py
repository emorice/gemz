"""
Information-driven Gaussian mixture models
"""

import numpy as np
import distrax
import jax.numpy as jnp

from gemz import jax_utils, linalg
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
            'reg_covar': np.ones(n_groups),
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

def _igmm_stats(data_np, responsibilities_p, reg_covar):
    """
    Compute average misfit and log det of precision for a single component

    Returns:
        triple of scalars, the weighted sum of the log-dets of the covariance
            matrices, the average misfit, and group size
    """
    precomp = gmm.precompute_loo(dict(
        data=data_np,
        reg_covar=reg_covar,
        responsibilities=responsibilities_p
        ))

    group_size = np.sum(responsibilities_p, -1)
    len1 = data_np.shape[0]

    means_pn = precomp['means']
    covariances_pnn = precomp['covariances']
    precisions_pnn = precomp['precisions']

    centered_data_pn = data_np.T - means_pn

    # sum over N (twice)
    misfits_p = np.sum(
        (precisions_pnn @ centered_data_pn[..., None])
        * centered_data_pn[..., None],
        (-2, -1)
        )

    # batched det over N x N
    _sign, ldet_covs_p = np.linalg.slogdet(covariances_pnn)
    total_ldet_cov = responsibilities_p @ ldet_covs_p

    avg_misfit = (
            np.sum(misfits_p * responsibilities_p, -1)
            / (len1 * group_size)
            )

    return total_ldet_cov, avg_misfit, group_size

@jaxify
def _igmm_obj(data, responsibilities, reg_covar, barrier_strength=0.1):
    """
    Information-GMM objective

    Args:
        data: N x P, P being the large dimension
        responsibilities: K x P, K being the number of groups
        reg_covar: K, the regularization for each group
    """

    responsibilities_kp = responsibilities
    reg_covar_k = reg_covar
    len1 = data.shape[0]

    total_ldet_covs_k, avg_misfits_k, group_sizes_k = linalg.imap(lambda resp_p, reg:
            _igmm_stats(data, resp_p, reg), responsibilities_kp, reg_covar_k)

    # scalar
    exp_log_lk = (
        - 0.5 * np.sum(total_ldet_covs_k)
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
