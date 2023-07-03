"""
Simple linear predictions.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.special as sc

from gemz import linalg
from gemz.model import (
        Model, Distribution, VstackTensorContainer,
        as_tensor_container, IndexTuple, as_index
        )

from . import methods
from .methods import ModelSpec
from .transformations import AddedConstantModel


# Interface V1
# ============

methods.add_module('linear', __name__)

def fit(data):
    """
    Computes a representation of the precision matrix of the first axis of the
    data.

    Args:
        data: N1 x N2, assuming N1 < N2
    Returns:
        precision: representation of a N2 x N2 precision matrix
    """

    # N1 x N1, the other axis precision
    dual_precision = np.linalg.pinv(data @ data.T, hermitian=True)

    precision = linalg.SymmetricLowRankUpdate(1., data.T, - dual_precision)

    return {
        'precision': precision
        }

def predict_loo(model, new_data):
    """
    Predicts each entry in new data from the others assuming the given model

    Args:
        model: the representation of a N2 x N2 precision matrix
        new_data: N1' x N2, where N2 matches the training data
    Returns:
        predictions: N1' x N2
    """
    precision = model['precision']

    # N1' x N2, linear predictions but not scaled and with the diagonal
    unscaled_residuals = new_data @ precision

    residuals = unscaled_residuals / np.diagonal(precision)

    predictions = new_data - residuals

    return predictions

def spectrum(data):
    """
    Estimated spectrum used implicitely by the linear interpolation

    Args:
        data: N1 x N2, assuming N1 < N2
    Returns:
        spectrum of the N2 covariance, of length N2, in descending order,
        including the N2 - N1 zeros at the end
    """
    len1, len2 = data.shape
    singular_values = np.linalg.svd(data, compute_uv=False)
    return np.hstack((
        singular_values ** 2 / len2,
        np.zeros(len2 - len1)
        ))

# Interface V2
# ============

def pinv_logdet(array):
    """
    Transposed pseudo inverse and square determinant of a rectangular matrix
    """
    # Adapted from the numpy implementation of pinv

    rcond = np.array(1e-15)
    u, s, vt = np.linalg.svd(array, full_matrices=False)

    # discard small singular values
    cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
    large = s > cutoff

    logdet = 2.* np.sum(np.log(s[large]))

    s = np.divide(1, s, where=large, out=s)
    s[~large] = 0

    pinvt = np.matmul(u * s[..., None, :], vt)

    return pinvt, logdet

def block_loo(indexes, data, dfs, ginv_function):
    """
    Generic implementation of block-loo conditional statistics

    Arguments:
        dfs: prior degrees of freedom
    """
    # Naming: n: observed rows, m: new rows, p: columns
    rows, _cols = indexes
    data_np = data[~rows, :]
    data_mp = data[rows, :]

    # Fixme: this isn't actually a mp-dimensional distribution, it's p
    # distinct m-dimensional distributions corresponding to p conditionals,
    # but the Distribution object currently does a poor job of representing that
    return FindMeANameDistribution(
            data_np, data_mp, ginv_function, dfs
            )

def log_norm_std(dfs, len_left, len_right):
    """
    Log normalization constant of a standard matrix-t
    """
    small = min(len_left, len_right)
    large = max(len_left, len_right)

    args = 0.5 * (dfs + np.arange(small))

    return (
        np.sum(sc.gammaln(args))
        + 0.5 * small * large * np.log(np.pi)
        - np.sum(sc.gammaln(args + 0.5*large))
        )

class FindMeANameDistribution(Distribution):
    def __init__(self, data_np, data_mp, ginv_function, dfs):

        # Data
        self.data_np = data_np
        self.data_mp = data_mp

        # Shapes
        self.n_obs_rows, self.n_cols = np.shape(data_np)
        self.n_new_rows, _ = np.shape(data_mp)
        self.n_rows = self.n_obs_rows + self.n_new_rows
        self.total_dims = self.n_new_rows
        self.dfs = dfs

        # Precomputation on observed data
        self.ginv_np, self.logdet_obs = ginv_function(data_np)
        # The standardized RSS using a non-blind covariance
        # Because of the leakage, it can't go over 1
        self.obs_leaky_rss_p = np.sum(self.ginv_np * data_np, 0)

        # Precomputation on all (joint) data
        data_ap = np.vstack((data_np, data_mp))
        ginv_ap, self.logdet_joint = ginv_function(data_ap)
        self.joint_leaky_rss_p = np.sum(ginv_ap * data_ap, 0)

    @property
    def mean(self):
        # The blind rss, plus 1
        onep_rss_p = 1. / (1. - self.obs_leaky_rss_p)

        mean_mp = (
            (self.data_mp @ self.data_np.T) @ self.ginv_np
            - self.data_mp * self.obs_leaky_rss_p
            ) * onep_rss_p
        return mean_mp

    @property
    def sf_radial_observed(self):
        """
        Lazy computation of survival function
        """
        # This avoids the use of non-differentiable primitives in the
        # precomputation part
        sf_beta_quantile_p = (1. - self.joint_leaky_rss_p) / (1. - self.obs_leaky_rss_p)
        sf_p = sc.betainc(
                .5 * (self.dfs + self.n_obs_rows + self.n_cols - 1),
                .5 * self.n_new_rows,
                sf_beta_quantile_p
                )
        return sf_p

    @property
    def logpdf_observed(self):
        # FIXME: this does not include the standard constant (which changes with dfs
        # and shape)
        logpdf_p = 0.5 * (
                + self.logdet_obs
                - self.logdet_joint
                + (self.dfs + self.n_rows + self.n_cols - 2) * np.log1p(- self.joint_leaky_rss_p)
                - (self.dfs + self.n_obs_rows + self.n_cols - 2) * np.log1p(- self.obs_leaky_rss_p)
                - log_norm_std(
                    self.dfs + self.n_obs_rows + self.n_cols - 1,
                    self.n_new_rows, 1
                    )
                )
        # NOT times n_cols, this is a batch of vectors distributions
        return logpdf_p

class LinearModel(Model):
    """
    Linear model, unregularized, without uncertainties
    """
    def _condition_block_block(self, unobserved_indexes, data):
        rows, cols = unobserved_indexes
        mean = (data[rows, ~cols] @ np.linalg.pinv(data[~rows, ~cols])
                @ data[~rows, cols])
        return Distribution(mean, total_dims=mean.size)

    def _condition_block_loo(self, unobserved_indexes, data):
        return block_loo(unobserved_indexes, data, 0, pinv_logdet)

def make_model(spec: ModelSpec):
    assert spec == {'model': 'linear'}
    return AddedConstantModel(LinearModel(), offset=1.0)


# Older reference implementations
# ===============================

def ref_fit(data):
    """
    Older reference implementation

    Computes precision matrix
    """
    # data: N x D

    # D x D
    prec = np.linalg.inv(data.T @ data)

    return {
        'precision': prec,
        'train': data,
        'transformed_train': data @ prec,
        }

def ref_predict_loo(model, new_data):
    """
    Older reference implementation

    Leave-one out prediction on new observations
    """
    base_prec = model['precision']
    train = model['train']
    t_train = model['transformed_train']

    base_covs = train.T @ new_data
    covs = base_covs[:, None] - train.T * new_data

    ## still biases by use of self in prec
    base_weights = base_prec @ covs
    weights = (
        base_weights
        + t_train.T
            * np.sum(t_train.T * covs, 0)
            / (1. - np.sum(train * t_train, 1))
        )

    preds = np.sum(train.T * weights, 0)
    return preds
