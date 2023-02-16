"""
Matrix-t utils
"""

import dataclasses as dc
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.scipy.special as jsc

@dataclass
class MatrixT:
    """
    Specification of a matrix-t distribution
    """
    dfs: float
    left: Any
    right: Any

    _: dc.KW_ONLY
    mean: Any = 0.

    def __post_init__(self):
        self.len_left = self.left.shape[-1]
        self.len_right = self.right.shape[-1]

@dataclass
class MatrixTObservation(MatrixT):
    """
    Observation from a matrix-t distribution of known parameters
    """
    observed: Any

def log_norm_std(dfs, len_left, len_right):
    """
    Log normalization constant of a standard matrix-t
    """
    small = min(len_left, len_right)
    large = max(len_left, len_right)

    args = 0.5 * (dfs + jnp.arange(small))

    return (
        jnp.sum(jsc.gammaln(args))
        + 0.5 * small * large * jnp.log(jnp.pi)
        - jnp.sum(jsc.gammaln(args + 0.5*large))
        )

def gen_matrix(mto):
    """
    Dense generating matrix
    """
    cdata = mto.observed - mto.mean

    return jnp.block(
        [[ mto.left, cdata ],
         [ (-cdata).T, mto.right ]]
        )

def ref_log_pdf(mto):
    """
    Reference log-pdf of a matrix-t observation

    Note: should be correctly normalized but yet to be tested.
    """
    _sign, logdet_left = jnp.linalg.slogdet(mto.left)
    _sign, logdet_right = jnp.linalg.slogdet(mto.right)

    _sign, logdet = jnp.linalg.slogdet(gen_matrix(mto))

    return (
            0.5 * (mto.dfs + mto.len_right - 1) * logdet_right
            + 0.5 * (mto.dfs + mto.len_left - 1) * logdet_left
            - log_norm_std(mto.dfs, mto.len_left, mto.len_right)
            - 0.5 * (mto.dfs + mto.len_left + mto.len_right - 1) * logdet
            )

@dataclass
class NonCentralMatrixT:
    """
    MatrixT with marginalized mean parameters
    """
    dfs: float
    left: Any
    right: Any
    scale: float
    scale_mean_left: float
    scale_mean_right: float

    _: dc.KW_ONLY
    mean: Any = 0.

    def __post_init__(self):
        self.len_left = self.left.shape[-1]
        self.len_right = self.right.shape[-1]

@dataclass
class NonCentralMatrixTObservation(NonCentralMatrixT):
    """
    Observation from a noncentral mt
    """
    observed: Any

def as_padded_mto(ncmto):
    """
    Generates the padded matrix t equivalent to a given non central matrix t

    Data is padded on top and right (first row, last column)
    """
    len_left, len_right = ncmto.len_left, ncmto.len_right
    scale = ncmto.scale

    padded_data = jnp.block(
            [[ jnp.ones(len_right), 0. ],
                [ ncmto.observed, jnp.ones(len_left)[:, None] ]]
            )
    padded_left = jnp.block(
            [[ ncmto.scale_mean_left/scale, jnp.zeros(len_left) ],
                [ jnp.zeros(len_left)[:, None], scale*ncmto.left ]]
            )
    padded_right = jnp.block(
            [[ scale*ncmto.right, jnp.zeros(len_right)[:, None] ],
             [ jnp.zeros(len_right), ncmto.scale_mean_right/scale ]]
            )

    return MatrixTObservation(ncmto.dfs, padded_left, padded_right, padded_data)

def ref_log_kernel_noncentral(ncmto):
    """
    Matrix-t with left and right means marginalized out
    """
    # While the pdf of the mto is well normalized, we're conditioning on a row
    # so the constant changes.
    return ref_log_pdf(as_padded_mto(ncmto))

def ref_uni_cond(mto: MatrixTObservation):
    """
    Conditional distributions of individual entries
    """
    igmat = jnp.linalg.inv(gen_matrix(mto))

    inv_diag = jnp.diagonal(igmat)
    inv_diag_left, inv_diag_right = inv_diag[:mto.len_left], inv_diag[mto.len_left:]
    inv_data = igmat[:mto.len_left, mto.len_left:]

    inv_diag_prod = inv_diag_left[:, None] * inv_diag_right[None, :]
    dets = inv_diag_prod + inv_data**2

    residuals = - inv_data / dets

    dfs = mto.dfs + (mto.len_left - 1) + (mto.len_right - 1)
    means = mto.observed - residuals
    variances = inv_diag_prod / ((dfs - 2.) * dets**2)
    logks = 0.5 * (dfs * jnp.log(inv_diag_prod) - (dfs - 1) * jnp.log(dets))
    logps = logks - jsc.betaln(0.5 * dfs, 0.5)
    return means, variances, logps

def ref_uni_cond_noncentral(ncmto):
    """
    Conditional distributions of individual entries
    """

    mto = as_padded_mto(ncmto)
    return tuple(stat[1:, :-1] for stat in ref_uni_cond(mto))

@dataclass
class Wishart:
    """
    Specification of a Wishart distribution
    """
    dfs: float
    gram: Any

def post_left(mto: MatrixT) -> Wishart:
    """
    Compute the posterior distribution of the left gram matrix after observing
    data
    """
    return Wishart(
        dfs=mto.dfs + mto.len_right,
        gram=mto.left + mto.observed @ (
            jnp.linalg.inv(mto.right) @ mto.observed.T
            )
        )

def from_left(wishart_left: Wishart, right, mean=0.) -> MatrixT:
    """
    Generate a matrix-t from an existing posterior left gram
    """
    return MatrixT(
            dfs=wishart_left.dfs,
            left=wishart_left.gram,
            right=right,
            mean=mean
            )

def observe(mtd: MatrixT, data) -> MatrixTObservation:
    """
    Pack distribution and data into an observation object
    """
    return MatrixTObservation(**dc.asdict(mtd), observed=data)
