"""
Information-driven Gaussian mixture models
"""

import numpy as np
from gemz import jax_utils
from gemz.jax_numpy import jaxify
from . import methods
from .gmm import GMM

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
            'responsibilities': resp0
            },
        data={
            'data': data,
            'barrier_strength': barrier_strength,
            },
        bijectors={
            'responsibilities': jax_utils.Softmax()
            },
        scipy_method='L-BFGS-B',
        )

    resps = max_results['opt']['responsibilities']
    groups = np.argmax(
        resps,
        axis=0
        )

    return GMM.precompute_loo({
        'data': data,
        'groups': groups,
        'responsibilities': resps
        })

predict_loo = GMM.predict_loo

# Objective functions
# ===================

@jaxify
def _igmm_obj(data, responsibilities, barrier_strength=0.1):
    """
    Information-GMM objective

    Args:
        data: N x P, P being the large dimension
        responsibilities: K x P, K being the number of groups
    """

    precomp = GMM.precompute_loo(dict(
        data=data,
        responsibilities=responsibilities
        ))

    # K x P x N, loo-sum over P
    means = precomp['means']

    # K x P x N X N
    covariances = precomp['covariances']
    precisions = precomp['precisions']

    # K x P x N
    centered_data = data.T - means

    # K x P, sum over N
    misfits = np.sum(
        (precisions @ centered_data[..., None])
        * centered_data[..., None],
        (-2, -1)
        )

    # K x P, det over N x N
    _signs, log_det_covs = np.linalg.slogdet(covariances)

    # scalar
    agg_misfits = np.tensordot(responsibilities, misfits)

    # scalar
    exp_log_lk = (
        - 0.5 * np.tensordot(responsibilities, log_det_covs)
        - 0.5 * agg_misfits
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
    precomp = GMM.precompute(dict(data=data, responsibilites=responsibilities))

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
