"""
Matrix-t utils
"""

import jax.numpy as jnp


def ref_log_kernel(data, dfs, left, right, mean):
    """
    Reference unnormalized log-likelihood of a matrix-t distribution
    """
    len_left, len_right = data.shape
    _sign, logdet_left = jnp.linalg.slogdet(left)
    _sign, logdet_right = jnp.linalg.slogdet(right)

    _sign, logdet = jnp.linalg.slogdet(jnp.block(
        [[ left, (data - mean) ],
         [ (mean - data).T, right ]]
        ))

    return (
            0.5 * len_left * logdet_right
            + 0.5 * len_right * logdet_left
            - 0.5 * (dfs + len_left + len_right - 1) * logdet
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
    return ref_log_kernel(padded_data, dfs, padded_left, padded_right, 0.)
