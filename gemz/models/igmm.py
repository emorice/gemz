"""
Information-driven Gaussian mixture models
"""

import numpy as np
from gemz import jax_utils
from gemz import linalg
from gemz.jax_numpy import jaxify

@jaxify
def igmm_obj(data, responsibilities, barrier_strength=0.1):
    """
    Information-GMM objective

    Args:
        data: N x P, P being the large dimension
        responsibilities: K x P, K being the number of groups
    """
    # K x P, loo-sum over P
    group_sizes = linalg.loo_sum(responsibilities, -1)

    # K x P x N, loo-sum over P
    means = linalg.loo_matmul(responsibilities, data.T) / group_sizes[..., None]

    # K x P x N x N implicit matrix stack, noncentered covariances
    grams = (
        linalg.loo_square(data, responsibilities)
        / group_sizes[..., None, None]
        )

    # K x P x N x N implicit matrix stack
    covariances = linalg.SymmetricLowRankUpdate(
        base=grams,
        # Extra contracting dim (outer product)
        factor=means[..., None],
        weight=-1,
        )

    # K x P x N X N
    precisions = np.linalg.inv(covariances)

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
def gmm_obj(data, responsibilities, barrier_strength=0.1):
    """
    Classical GMM objective for reference.

    Args:
        data: N x P, P being the large dimension
        responsibilities: K x P, K being the number of groups
    """
    # K, sum over P
    group_sizes = np.sum(responsibilities, -1)

    # K x N, sum over P
    means = responsibilities @ data.T / group_sizes[:, None]
    # K x N x N. O(KNP) intermediate, may be improved
    covariances = (
        (data * responsibilities[:, None, :]) @ data.T
            / group_sizes[:, None, None]
        - means[:, :, None] * means[:, None, :]
        ) #+ np.diag(np.ones_like(means[0])) * .5
    # K x N X N
    precisions = np.linalg.inv(covariances)

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
    _signs, log_det_covs = np.linalg.slogdet(covariances)

    agg_misfits = np.tensordot(responsibilities, misfits)
    # scalar
    exp_log_lk = (
        - 0.5 * group_sizes @ log_det_covs
        - 0.5 * agg_misfits
        )
    # scalar
    entropy = - np.tensordot(responsibilities, np.log(responsibilities))

    # scalar
    barrier = barrier_strength * np.sum(np.log(responsibilities))

    return exp_log_lk + entropy + barrier

def fit(data, n_groups, seed=0, barrier_strength=1e-2, init_resps=None):
    """
    Learns a GMM with an information loss

    Args:
        data: len1 x len2, where the len1 rows will be clustered.
    """

    if init_resps is None:
        # Random uniform initialization
        rng = np.random.default_rng(seed)
        resp0 = rng.uniform(size=(n_groups, data.shape[0]))
        resp0 /= resp0.sum(axis=0)
    else:
        resp0 = init_resps

    max_results = jax_utils.maximize(
        igmm_obj,
        init={
            'responsibilities': resp0
            },
        data={
            'data': data.T,
            'barrier_strength': barrier_strength,
            },
        bijectors={
            'responsibilities': jax_utils.Softmax()
            },
        scipy_method='L-BFGS-B',
        #has_aux=True
        )

    print(max_results)

    resps = max_results['opt']['responsibilities']
    groups = np.argmax(
        resps,
        axis=0
        )

    return {
        'groups': groups,
        'responsibilities': resps
        }
