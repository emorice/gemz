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

def fit_eval(model_spec, data_fold, loss_name, ops=methods):
    """
    Compound fit and eval_loss call

    Returns:
        dict with keys:
            'fit': the model fit, type depending on model
            'loss': the loss value on the given data split
    """

    fitted = ops.fit(model_spec, data_fold['train'])
    loss = ops.eval_loss(model_spec, fitted, data_fold['test'], loss_name)
    return { 'fit': fitted, 'loss': loss }

def cv_fit_eval(model_spec, data, fold_count, loss_name, ops=methods):
    """
    Fits and eval a model on each fold of the data and return the total loss

    Returns:
        a dict with keys:
            'folds': list of per-fold results, each a dict with keys:
                'data': the train and test data in a two-keys dict
                'fitted', 'loss': the fitted model and loss value
            'loss': the total loss
    """
    folds = [
            { 'data': {'train': train, 'test': test}}
            for train, test in [
                ops.fold(data, i, fold_count)
                for i in range(fold_count) ]
            ]
    for _fold in folds:
        _fold.update(
            fit_eval(model_spec, _fold['data'], loss_name, ops=ops)
            )
    total_loss = ops.aggregate_losses([_fold['loss'] for _fold in folds])

    return {'folds': folds, 'loss': total_loss}
