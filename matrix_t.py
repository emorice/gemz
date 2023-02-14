"""
Matrix-t utils
"""


from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

@dataclass
class MatrixT:
    """
    Specification of a Matrix-t distribution
    """
    observed: Any
    dfs: float
    left: Any
    right: Any
    mean: Any = 0.

    def __post_init__(self):
        self.len_left = self.left.shape[-1]
        self.len_right = self.right.shape[-1]

def gen_matrix(mtd):
    """
    Dense generating matrix
    """
    cdata = mtd.observed - mtd.mean

    return jnp.block(
        [[ mtd.left, cdata ],
         [ (-cdata).T, mtd.right ]]
        )

def ref_log_kernel(mtd):
    """
    Reference unnormalized log-likelihood of a matrix-t distribution
    """
    _sign, logdet_left = jnp.linalg.slogdet(mtd.left)
    _sign, logdet_right = jnp.linalg.slogdet(mtd.right)

    _sign, logdet = jnp.linalg.slogdet(gen_matrix(mtd))
    return (
            0.5 * mtd.len_left * logdet_right
            + 0.5 * mtd.len_right * logdet_left
            - 0.5 * (mtd.dfs + mtd.len_left + mtd.len_right - 1) * logdet
            )

@dataclass
class NonCentralMatrixT:
    """
    MatrixT with marginalized mean parameters
    """
    observed: Any
    dfs: float
    left: Any
    right: Any
    scale: float
    scale_mean_left: float
    scale_mean_right: float
    mean: Any = 0.

    def __post_init__(self):
        self.len_left = self.left.shape[-1]
        self.len_right = self.right.shape[-1]

def as_padded_mtd(ncmtd):
    """
    Generates the padded matrix t equivalent to a given non central matrix t

    Data is padded on top and right (first row, last column)
    """
    len_left, len_right = ncmtd.len_left, ncmtd.len_right
    scale = ncmtd.scale

    padded_data = jnp.block(
            [[ jnp.ones(len_right), 0. ],
                [ ncmtd.observed, jnp.ones(len_left)[:, None] ]]
            )
    padded_left = jnp.block(
            [[ ncmtd.scale_mean_left/scale, jnp.zeros(len_left) ],
                [ jnp.zeros(len_left)[:, None], scale*ncmtd.left ]]
            )
    padded_right = jnp.block(
            [[ scale*ncmtd.right, jnp.zeros(len_right)[:, None] ],
             [ jnp.zeros(len_right), ncmtd.scale_mean_right/scale ]]
            )

    return MatrixT(padded_data, ncmtd.dfs, padded_left, padded_right)

def ref_log_kernel_noncentral(ncmtd):
    """
    Matrix-t with left and right means marginalized out
    """
    return ref_log_kernel(as_padded_mtd(ncmtd))

def ref_uni_cond(mtd):
    """
    Conditional means of individual entries
    """
    igmat = jnp.linalg.inv(gen_matrix(mtd))

    inv_diag = jnp.diagonal(igmat)
    inv_diag_left, inv_diag_right = inv_diag[:mtd.len_left], inv_diag[mtd.len_left:]
    inv_data = igmat[:mtd.len_left, mtd.len_left:]

    inv_diag_prod = inv_diag_left[:, None] * inv_diag_right[None, :]
    dets =inv_diag_prod + inv_data**2

    residuals = - inv_data / dets

    dfs = mtd.dfs + (mtd.len_left - 1) + (mtd.len_right - 1)
    means = mtd.observed - residuals
    variances = inv_diag_prod / ((dfs - 2.) * dets**2)
    return means, variances

def ref_uni_cond_noncentral(ncmtd):
    """
    Conditional means of individual entries
    """

    mtd = as_padded_mtd(ncmtd)
    padded_means, padded_vars = ref_uni_cond(mtd)

    return padded_means[1:, :-1], padded_vars[1:, :-1]
