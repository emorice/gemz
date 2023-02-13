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

def ref_log_kernel_centered(data, dfs, left, right, scale, rel_var_mean_left,
        rel_var_mean_right):
    """
    Matrix-t with left and right means marginalized out
    """
    len_left, len_right = data.shape
    padded_data = jnp.block(
            [[ jnp.ones(len_right), 0. ],
                [ data, jnp.ones(len_left)[:, None] ]]
            )
    padded_left = jnp.block(
            [[ rel_var_mean_left/scale, jnp.zeros(len_left) ],
                [ jnp.zeros(len_left)[:, None], scale*left ]]
            )
    padded_right = jnp.block(
            [[ scale*right, jnp.zeros(len_right)[:, None] ],
             [ jnp.zeros(len_right), rel_var_mean_right/scale ]]
            )
    return ref_log_kernel(MatrixT(padded_data, dfs, padded_left, padded_right))
