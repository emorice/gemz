"""
Module containing all statistical model fit and prediction code
"""
import sys

from . import (
    kmeans, linear, wishart,
    linear_shrinkage,
    linear_shrinkage_cv,
    lscv_precision_target,
    lscv_free_diagonal,
    nonlinear_shrinkage,
    cmk,
    gmm, igmm
    )

from . import methods
from .methods import (
        get,
        fit, predict_loo, eval_loss,
        fold, aggregate_losses
        )

def fit_eval(model_spec, train_data, test_data, loss_name, ops=methods):
    """
    Compound fit and eval_loss call
    """
    fitted = ops.fit(model_spec, train_data)
    loss = ops.eval_loss(model_spec, fitted, test_data, loss_name)
    return fitted, loss

def cv_fit_eval(model_spec, data, fold_count, loss_name, ops=methods):
    """
    Fits and eval a model on each fold of the data and return the total loss
    """
    folds = [ ops.fold(data, i, fold_count) for i in range(fold_count) ]
    fit_evals = [
        fit_eval(model_spec, *_fold, loss_name, ops=ops)
        for _fold in folds
        ]

    return ops.aggregate_losses([loss for _fitted, loss in fit_evals])
