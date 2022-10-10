"""
Regularized linear model obtained by keeping only some factors of a singular
value decompistion
"""

import numpy as np

from gemz import linalg
from .methods import add_module
from . import svd

try:
    import pmbio_peer
    HAS_PEER = True
except ModuleNotFoundError:
    HAS_PEER = False

add_module('peer', __name__)

def fit(data, n_factors):
    """
    Builds a representation of the precision matrix by using `n_factors`
    inferred PEER factors
    """

    if not HAS_PEER:
        raise RuntimeError('The module pmbio_peer could not be loaded')


    # Define model
    model = pmbio_peer.PEER()

    # "The data matrix is assumed to have N rows and G columns, where N is the
    # number of samples, and G is the number of genes."
    model.setNk(n_factors)
    model.setPhenoMean(data)

    model.setPriorAlpha(0.001, 0.01)
    model.setPriorEps(0.1, 10.)
    model.setNmax_iterations(1000)

    # Run it
    model.update()

    # Extract relevant parameters
    cofactors_gk = model.getW()
    factors_nk = model.getX()
    precisions_g = np.squeeze(model.getEps())

    print(model.getAlpha())

    # Build covariance
    cov = linalg.SymmetricLowRankUpdate(
            1. / precisions_g,
            cofactors_gk @ factors_nk.T / np.sqrt(data.shape[0]),
            1.
            )

    return {
        'precision': np.linalg.inv(cov)
        }

# All other methods are inherited from the SVD case
predict_loo = svd.predict_loo
get_name = svd.get_name
make_grid = svd.make_grid
get_grid_axis = svd.get_grid_axis
