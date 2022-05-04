"""
Simple linear predictions.
"""

import numpy as np

from gemz import linalg

def fit(data):
    """
    Computes a representation of the precision matrix of the first axis of the
    data.

    Args:
        data: N1 x N2, assuming N1 > N2
    Returns:
        precision: representation of a N1 x N1 precision matrix
    """

    # N2 x N2, the other axis precision
    dual_precision = np.linalg.inv(data.T @ data)

    precision = linalg.RWSS(1., data, - dual_precision)

    return {
        'precision': precision
        }

def predict_loo(model, new_data):
    """
    Predicts each entry in new data from the others assuming the given model

    Args:
        model: the representation of a N1 x N1 precision matrix
        new_data: N1 x N'2, where N1 matches the training data
    Returns:
        predictions: N1 x N'2
    """
    new_data = np.atleast_2d(new_data.T).T

    precision = model['precision']

    # N1 x N'2, linear predictions but not scaled and with the diagonal
    unscaled_residuals = precision @ new_data

    residuals = unscaled_residuals / np.diagonal(precision)[:, None]

    predictions = new_data - residuals

    return predictions


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
