"""
Regularized linear predictions using Wishart priors
"""

import numpy as np
import jax
import jax.numpy as jnp

from gemz import jax_utils
from . import methods

@methods.add('wishart')
class Wishart:
    """
    Wishart evidence based linear shrinkage
    """

    @staticmethod
    def fit(data):
        """
        Computes the regularized precision matrix

        Args:
            data: N1 x N2, assuming N1 < N2

        Returns:
            Optimal regularizer of the N1 x N1 covariance matrix
        """

        # N1 x N1
        spectrum, _ = np.linalg.eigh(data @ data.T)

        opt = jax_utils.maximize(
            wishart_marginal_likelihood,
            init=dict(
                prior_edf_ln=0.,
                prior_var_ln=0.,
                ),
            data=dict(
                data_cov_spectrum=spectrum,
                n_samples=data.shape[1], # N2
                n_dims=data.shape[0] # N1
                )
            )

        return {**opt, 'train': data}

    @staticmethod
    def predict_loo(model, new_data):
        """
        Leave-one out prediction on new observations

        Args:
            new_data: N1' x N2
        """

        # This was written for one test observation only, and a pain to batch so
        # wrapped in a loop for now.
        preds = []
        for new in new_data:

            # N1 x N2
            train = model['train']
            # 1+N1 x N2
            joint = np.vstack((new[None, :], train))

            # 1+N1 x 1+N1
            reg_cov = joint @ joint.T + np.exp(model['opt']['prior_var_ln'])

            # 1+N1 x 1+N1
            base_prec = np.linalg.inv(reg_cov)

            # N2
            base_raw_preds = - base_prec[0, 1:] @ train
            # 1
            base_scale = base_prec[0, 0]

            # 1+N1 x N2
            trans = base_prec @ joint

            # N2
            r1fs = 1. + np.sum(trans * joint, 0)

            # N2
            raw_preds = (
                base_raw_preds
                - trans[0, :] / r1fs * jnp.sum(trans[1:, :] * train, 0)
                )

            # N2
            scale = (
                base_scale
                + trans[0, :]**2 / r1fs
                )

            preds.append(raw_preds / scale)

        return np.vstack(preds)

def wishart_marginal_likelihood(data_cov_spectrum, n_samples, n_dims, prior_edf_ln, prior_var_ln):
    """
    Evidence of integrated gaussian likelihood against a Wishart
    identity-proportional prior.

    Args:
        data_cov_sepctrum: spectum of the empirical covariance matrix
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
