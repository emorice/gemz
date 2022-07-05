"""
Information-driven Gaussian mixture models
"""

import numpy as np
from gemz import jax_utils

def igmm_obj(data, responsibilities):
    """
    Information-based objective function for GMM
    """
    return 0.

def fit(data, n_groups, seed=0):
    """
    Learns a GMM with an information loss

    Args:
        data: len1 x len2, where the len1 rows will be clustered.
    """

    # Random uniform initialization
    rng = np.random.default_rng(seed)
    resp0 = rng.uniform(size=(n_groups, data.shape[0]))
    resp0/= resp0.sum(axis=0)

    max_results = jax_utils.maximize(
        igmm_obj,
        init={
            'responsibilities': resp0
            },
        data={
            'data': data
            }
        )

    groups = np.argmax(
        max_results['opt']['responsibilities'],
        axis=0
        )

    return {
        'groups': groups
        }
