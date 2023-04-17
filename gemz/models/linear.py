"""
Simple linear predictions.
"""

import numpy as np

from gemz import linalg
from . import methods
from .methods import ModelSpec
from gemz.model import Model, Conditioner


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

def make_model(spec: ModelSpec, conditioner: Conditioner):
    """
    Interface entry point
    """
    return LinearModel(spec, conditioner)

class LinearModel(Model):
    """
    Linear model, unregularized
    """

    def mean(self, *data, **kwdata):
        return 0.

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
