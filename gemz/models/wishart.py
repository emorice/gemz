"""
Regularized linear predictions using Wishart priors
"""

import numpy as np
import jax
import jax.numpy as jnp

from gemz import jax_utils

def fit(data):
    """
    Computes the regularized precision matrix
    """
    # data: N x D

    # D x D
    spectrum, _ = np.linalg.eigh(data.T @ data)

    opt = jax_utils.maximize(
        wishart_marginal_likelihood,
        init=dict(
            prior_edf_ln=0.,
            prior_var_ln=0.,
            ),
        data=dict(
            data_cov_spectrum=spectrum,
            n_samples = data.shape[0],
            n_dims=data.shape[1]
            )
        )


    return opt

def wishart_marginal_likelihood(data_cov_spectrum, n_samples, n_dims, prior_edf_ln, prior_var_ln):
    """
    Evidence of integrated gaussian likelihood against a Wishart
    identity-proportional prior.

    Args:
        data_cov_chol_diag: diagonal of the cholesky factorization of the
            empirical covariance matrix
        prior_edf_ln: log of the prior degrees of freedom in excess of n_dims
    """
    prior_df = n_dims + jnp.exp(prior_edf_ln)

    log_ev = 0.

    # Constant
    log_ev += 0.5 * n_samples * n_dims * jnp.log(jnp.pi)

    # Prior multigamma
    log_ev -= jax.scipy.special.multigammaln(0.5 * prior_df, n_dims)

    # Posterior multigamma
    log_ev += jax.scipy.special.multigammaln(
        0.5 * (prior_df + n_samples),
        n_dims
        )

    # Prior determinant
    log_ev += 0.5 * prior_df * n_dims * prior_var_ln

    # Posterior determinant
    log_ev -= 0.5 * (prior_df + n_samples) * jnp.sum(
        jnp.log(data_cov_spectrum + jnp.exp(prior_var_ln))
        )

    return log_ev

def predict_loo(model, new_data):
    """
    Leave-one out prediction from existing clusters on new observations
    """
    return np.full_like(new_data, np.nan)
