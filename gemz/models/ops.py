"""
Meta-model building blocks
"""

import sys
import numpy as np
import logging

from . import methods, cv

logger = logging.getLogger('gemz')

# Basic ops
# =========

def predict_loo(model_spec, model_fit, test_data):
    """
    Like `fit` for the `predict_loo` method
    """
    return methods.get(model_spec['model']).predict_loo(model_fit, test_data)

def eval_loss(model_spec, model_fit, test_data, loss_name):
    """
    Simple wrapper around the losses in `cv`.

    Factored out as its own function to make it easy to pipeline
    """
    loss_fn = cv.LOSSES[loss_name]

    model = methods.get(model_spec['model'])

    return loss_fn(model, model_fit, test_data)

def fold(data, fold_index, fold_count, seed=0):
    """
    Generate a split of the data along its first axis

    Returns:
        train, test, train_mask
    """
    len1, *_ = data.shape

    rng = np.random.default_rng(seed)
    random_rank = rng.choice(len1, len1, replace=False)

    in_fold = random_rank % fold_count != fold_index

    return data[in_fold, ...], data[~in_fold, ...], in_fold

def aggregate_losses(losses):
    """
    Trivial total loss computation
    """
    return sum(losses)

def select_best(grid):
    """
    Trivial selection of model with smallest loss
    """
    best_model, _results = grid[0]
    best_loss = _results['loss']

    for model, results in grid:
        loss = results['loss']
        if loss < best_loss:
            best_model, best_loss = model, loss

    return best_model

def aggregate_residuals(data, predictions):
    """
    Aggregates partial predictions on subsets of a dataset

    Args:
        predictions: list of tuple (mask, array) with the array containing
            predictions, and the mask being a one dimensional boolean mask
            saying which rows of a virtual larger arrays this partial array maps
            to, with true entries marking rows to skip.
    """
    res = np.array(data, copy=True)
    for mask, prediction in predictions:
        res[~mask, ...] -= prediction
    return res

# Meta-op ops
# ===========

_self = sys.modules[__name__]

def fit(model_spec, train_data, _ops=_self):
    """
    Fit a model from a model specification.

    Args:
        model_spec: a dictionnary containing the name of the model in 'model',
            and keyword arguments to pass along to the fit function of
            said model
        train_data: data to pass to fit
    """

    model = methods.get(model_spec['model'])
    kwargs = dict(model_spec)
    del kwargs['model']

    if hasattr(model, 'OPS_AWARE') and model.OPS_AWARE:
        kwargs['_ops'] = _ops

    logger.info('Fitting %s', methods.get_name(model_spec))

    return model.fit(train_data, **kwargs)

def cv_residualize(model_spec, data, fold_count=10, seed=0, _ops=_self):
    """
    Train and predict on the whole data in folds, returning residuals
    """
    predictions = []
    for i in range(fold_count):
        train, test, is_train = _ops.fold(data, i, fold_count, seed=seed)
        fitted = _ops.fit(model_spec, train)
        predictions.append((
            is_train, _ops.predict_loo(model_spec, fitted, test)
            ))

    return _ops.aggregate_residuals(data, predictions)

def build_eval_grid(inner, data, fold_count, loss_name, grid_size, grid_max, grid, seed, _ops=_self):
    """
    Generate a grid of models, then eval them through CV
    """
    inner_model = methods.get(inner['model'])

    if grid is None:
        grid = inner_model.cv.make_grid(data, grid_size=grid_size, grid_max=grid_max)

    specs = inner_model.cv.make_grid_specs(inner, grid)

    results = [
        _ops.cv_fit_eval(spec, data, fold_count, loss_name, seed, _ops=_ops)
        for spec in specs
        ]

    return list(zip(specs, results))

# Pure meta-ops
# =============

def fit_eval(model_spec, data_fold, loss_name, _ops=_self):
    """
    Compound fit and eval_loss call

    Args:
        model_spec: dict as expected by the 'fit' op.
        data_fold: dict with keys 'train' and 'test'
    Returns:
        dict with keys:
            'fit': the model fit, type depending on model
            'loss': the loss value on the given data split
    """

    fitted = _ops.fit(model_spec, data_fold['train'])
    loss = _ops.eval_loss(model_spec, fitted, data_fold['test'], loss_name)
    return { 'fit': fitted, 'loss': loss }

def cv_fit_eval(model_spec, data, fold_count=10, loss_name='RSS', seed=0, _ops=_self):
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
            for train, test, _ in [
                _ops.fold(data, i, fold_count, seed=seed)
                for i in range(fold_count) ]
            ]
    for _fold in folds:
        _fold.update(
            fit_eval(model_spec, _fold['data'], loss_name, _ops=_ops)
            )
    total_loss = _ops.aggregate_losses([_fold['loss'] for _fold in folds])

    return {'folds': folds, 'loss': total_loss}
