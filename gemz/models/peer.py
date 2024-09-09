"""
Regularized linear model obtained by keeping only some factors of a singular
value decompistion
"""

import sys
import numpy as np

from gemz import linalg
from .methods import add_module
from . import svd
from tqdm.auto import tqdm

try:
    import pmbio_peer
    pmbio_peer.peer.setVerbose(0)
    HAS_PEER = True
except ModuleNotFoundError:
    HAS_PEER = False

add_module('peer', __name__)

def fit(data, n_factors, reestimate_precision=False, custom_covariance=True, verbose=True):
    """
    Builds a representation of the precision matrix by using `n_factors`
    inferred PEER factors.

    Args:
        data: N1 x N2 matrix to factor, expecting N1 < N2
        n_factors: number of peer factors to learn.
        restimate_precision: whether to use the per-gene precision parameters
            learned by PEER for inference (default, False), or recalculate a
            simple estimate (True).
        custom_covariance: whether to assume loadings with covariance learned
            from the PEER loadings (default, True), or use loadings from the
            standard prior (False).
    """

    if not HAS_PEER:
        raise RuntimeError('The module pmbio_peer could not be loaded')

    n_iter = 1000

    if verbose:
        progress = tqdm(desc=get_name({'model': 'peer', 'n_factors': n_factors,
            'restimate_precision': reestimate_precision}), total=n_iter)

    # Define model
    model = pmbio_peer.PEER()

    # "The data matrix is assumed to have N rows and G columns, where N is the
    # number of samples, and G is the number of genes."
    model.setNk(n_factors)
    model.setPhenoMean(data)

    model.setPriorAlpha(0.001, 0.01)
    model.setPriorEps(0.1, 10.)

    tolerance = model.getTolerance()
    var_tolerance = model.getVarTolerance()

    # Convergence logic ported from cpp so that we can manually step and stop
    # the loop
    last_bound = -np.inf
    last_residual_var = np.inf
    current_bound = -np.inf
    current_residual_var = -np.inf
    delta_bound = np.inf
    delta_residual_var = np.inf

    # Run it for one iteration at once
    model.setNmax_iterations(1)
    for i in range(n_iter):
        model.update()

        if verbose:
            progress.update(1)

        last_bound = current_bound
        last_residual_var = current_residual_var
        current_bound = model.calcBound()
        current_residual_var = np.mean(model.getResiduals() ** 2)
        delta_bound = current_bound - last_bound
        delta_residual_var = last_residual_var - current_residual_var

        if (abs(delta_bound) < tolerance
                or abs(delta_residual_var) < var_tolerance):
            break

    if verbose:
        progress.close()
        if i + 1 < n_iter:
            print(f'Converged after {i + 1} iterations', file=sys.stderr, flush=True)

    # Extract relevant parameters
    cofactors_gk = model.getW()
    factors_nk = model.getX()

    if reestimate_precision:
        variances_g = np.var(model.getResiduals(), axis=0)
    else:
        variances_g = 1. / np.squeeze(model.getEps())

    if custom_covariance:
        root_signal_cov = cofactors_gk @ factors_nk.T / np.sqrt(data.shape[0])
    else:
        root_signal_cov = cofactors_gk

    # Build covariance
    cov = linalg.SymmetricLowRankUpdate(
            variances_g,
            root_signal_cov,
            1.
            )

    return {
        'precision': np.linalg.inv(cov)
        }

# All other methods are inherited from the SVD case
predict_loo = svd.predict_loo

def get_name(spec):
    """
    Printable short name
    """
    return svd.get_name(spec) + (
            'r' if spec.get('reestimate_precision')
            else ''
            ) + (
            'p' if not spec.get('custom_covariance', True)
            else ''
            )

cv = svd.cv
