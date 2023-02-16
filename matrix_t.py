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
class NonCentralMatrixT(MatrixT):
    """
    MatrixT with marginalized mean parameters

    Value "None" for any mean gram is treated as infinity, i.e., ignored and
    equivalent to the central case
    """
    gram_mean_left: float
    gram_mean_right: float

@dataclass
class NonCentralMatrixTObservation(NonCentralMatrixT):
    """
    Observation from a noncentral mt
    """
    observed: Any

def as_padded_mto(ncmto: NonCentralMatrixTObservation):
    """
    Generates the padded matrix t equivalent to a given non central matrix t

    Data is padded on top and right (first row, last column)
    """
    len_left, len_right = ncmto.len_left, ncmto.len_right

    if ncmto.gram_mean_left is None:
        if ncmto.gram_mean_right is None:
            # Pad nothing
            padded_data = ncmto.observed
        else:
            # Pad only right
            padded_data = jnp.hstack((
                ncmto.observed, jnp.ones(len_left)[:, None]
                ))
    else:
        if ncmto.gram_mean_right is None:
            # Pad only left
            padded_data = jnp.vstack((
                jnp.ones(len_right),
                ncmto.observed
                ))
        else:
            # Pad both. Mind the corner 0 !
            padded_data = jnp.block([
                [ jnp.ones(len_right),  0.                          ],
                [ ncmto.observed,       jnp.ones(len_left)[:, None] ]
                ])

    if ncmto.gram_mean_left is None:
        padded_left = ncmto.left
    else:
        padded_left = jnp.block(
                [[ ncmto.gram_mean_left, jnp.zeros(len_left) ],
                    [ jnp.zeros(len_left)[:, None], ncmto.left ]]
                )

    if ncmto.gram_mean_right is None:
        padded_right = ncmto.right
    else:
        padded_right = jnp.block(
                [[ ncmto.right, jnp.zeros(len_right)[:, None] ],
                 [ jnp.zeros(len_right), ncmto.gram_mean_right ]]
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

@dataclass
class Wishart:
    """
    Specification of a Wishart distribution
    """
    dfs: float
    gram: Any

def post_left(mto: MatrixTObservation) -> Wishart:
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

def nc_post_left(ncmto: NonCentralMatrixTObservation) -> Wishart:
    """
    Padded posterior Wishart.

    The right pseudo data gets included as a sample if present, without
    affecting dimension.
    The left pseudo data becomes an extra dimension at the beginning if present.
    """
    mto = as_padded_mto(ncmto)
    return post_left(mto)

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

def observe_left_padded(mtd: MatrixT, data):
    """
    Pack distribution and data into an observation object after padding data on
    top
    """
    return MatrixTObservation(
            **dc.asdict(mtd),
            observed=jnp.vstack((
                jnp.ones_like(data[0]),
                data
                ))
            )
