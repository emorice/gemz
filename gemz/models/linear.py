"""
Simple linear predictions.
"""

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

    ginv_np, logdet_obs = ginv_function(data_np)
    # The standardized RSS using a non-blind covariance
    # Because of the leakage, it can't go over 1
    obs_leaky_rss_p = np.sum(ginv_np * data_np, 0)

    # The blind rss, plus 1
    onep_rss_p = 1. / (1. - obs_leaky_rss_p)

    # Mean
    mean_mp = (
        (data_mp @ data_np.T) @ ginv_np
        - data_mp * obs_leaky_rss_p
        ) * onep_rss_p



    n_obs_rows, n_cols = np.shape(data_np)
    n_new_rows, _ = np.shape(data_mp)

    # SF

    data_ap = np.vstack((data_np, data_mp))
    ginv_ap, logdet_joint = ginv_function(data_ap)
    joint_leaky_rss_p = np.sum(ginv_ap * data_ap, 0)

    sf_beta_quantile_p = (1. - joint_leaky_rss_p) / (1. - obs_leaky_rss_p)
    sf_p = sc.betainc(
            .5 * (dfs + n_obs_rows + n_cols - 1),
            .5 * n_new_rows,
            sf_beta_quantile_p
            )

    # PDF
    # FIXME: this does not include the standard constant (which changes with dfs
    # and shape)
    n_rows = n_obs_rows + n_new_rows
    logpdf_p = 0.5 * (
            + logdet_obs
            - logdet_joint
            + (dfs + n_rows + n_cols - 2) * np.log1p(- joint_leaky_rss_p)
            - (dfs + n_obs_rows + n_cols - 2) * np.log1p(- obs_leaky_rss_p)
            )

    # Fixme: this isn't actually a mp-dimensional distribution, it's p
    # distinct m-dimensional distributions corresponding to p conditionals,
    # but the Distribution object currently does a poor job of representing that
    return Distribution(
            mean=mean_mp,
            sf_radial_observed=sf_p,
            logpdf_observed=logpdf_p,
            # NOT times n_cols, this is a batch of vectors distributions
            total_dims=n_new_rows,
            )

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
    return AddedConstantModel(LinearModel())


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
