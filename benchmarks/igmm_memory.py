"""
Memory profiling of the igmm objective function
"""

import numpy as np
import jax
from memory_profiler import profile

from gemz.models import gmm, igmm
from gemz.jax_numpy import jaxify
from gemz import linalg

def main():
    """
    Entry point
    """

    dims = (200, 500)
    n_groups = 180

    data = np.ones(dims) + np.random.default_rng(0).normal(size=dims)
    resps = np.ones((n_groups, dims[-1])) / n_groups

    k, p, n = n_groups, dims[1], dims[0]
    print('Data sizes')
    print(f'    PN  {p*n*8 / 2**20:.1f} MiB')
    print(f'    KNN {k*n*n*8 / 2**20:.1f} MiB')
    print(f'    KPN {k*p*n*8 / 2**20:.1f} MiB')

    #gmm.precompute_loo = profile(gmm.precompute_loo)
    linalg.eager_map = profile(linalg.eager_map)
    #igmm._igmm_stats = profile(igmm._igmm_stats)
    #linalg.SymmetricLowRankUpdate.inv = profile(linalg.SymmetricLowRankUpdate.inv)
    jax.value_and_grad(lambda r: igmm._igmm_obj(**{'data': data, 'responsibilities': r,
        'reg_covar': 0.01 * np.ones(n_groups)}))(resps)

if __name__ == '__main__':
    main()
