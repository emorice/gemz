"""
Meta-model building blocks
"""

import sys
import logging
from typing import TypedDict

from deprecation import deprecated

import numpy as np
from numpy.typing import NDArray

from gemz.model import (
        Model, VstackTensorContainer,
        IndexTuple, EachIndex, metric,
        MODULES
        ) # some reexports
from . import methods, cv
from .methods import ModelSpec, get_name

logger = logging.getLogger('gemz')

# Basic ops
# =========

def predict_loo(model_spec, model_fit, test_data):
    """
    Like `fit` for the `predict_loo` method
    """
    model = methods.get(model_spec['model'])
    item = getattr(model, 'EVAL_REQUIRES', None)
    if item:
        eval_fit = model_fit[item]
    else:
        eval_fit = model_fit
    return methods.get(model_spec['model']).predict_loo(eval_fit, test_data)

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

def extract_cv(grid):
    """
    Gather only the cross-validation data necessary for decision and plotting

    (The sum of all the cross validation models is commonly too large for memory)

    Args:
        t_grid: the grid task
    """
    # When not using galp this is trivial
    return s_extract_cv_losses(grid)

def s_extract_cv_losses(grid):
    """
    Extract only the loss item from the fits in the grid
    """
    return [(spec, {'loss': cfe['loss']}) for spec, cfe in grid]

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

class FoldDict(TypedDict):
    """
    A split of a data matrix into a train and test set
    """
    train: NDArray
    test: NDArray

def fit_eval(model_spec: ModelSpec, data_fold: FoldDict, loss_name: str,
        _ops=_self, verbose=True):
    """
    Compound fit and eval_loss call

    Args:
        model_spec: dict as expected by the 'fit' op.
        data_fold: dict with keys 'train' and 'test'
    Returns:
        dict with keys:
            'loss': the loss value on the given data split
    """

    if verbose:
        print(f'Fitting model: {get_name(model_spec)} {model_spec}', flush=True)

    model_name: str = model_spec['model']

    # Classic interface
    # -----------------
    if model_name not in MODULES:
        fitted = _ops.fit(model_spec, data_fold['train'])

        # Allows to only pass a part if the fit result, quickfix for cv
        # requiring too much memory
        model = methods.get(model_spec['model'])
        item = getattr(model, 'EVAL_REQUIRES', None)
        if item:
            eval_fit = fitted[item]
        else:
            eval_fit = fitted

        loss = _ops.eval_loss(model_spec, eval_fit, data_fold['test'], loss_name)
        # Note: when the two items are tasks, indexing the result of fit-eval
        # will safely give you a reference that can be used to get the loss
        # without loading the whole fitted model from disk. That is, extracting
        # just the loss from many models should be effcient.
        return {'fit': fitted, 'loss': loss}

    # Experimental interface
    # ----------------------
    model_obj = _ops.Model.from_spec(model_spec)
    data = VstackTensorContainer((
        data_fold['train'], data_fold['test']
        ))
    # No rows of train, all rows of test
    test_rows = IndexTuple((slice(0, 0), slice(None)))
    test_cols = EachIndex
    conditional = model_obj._condition((test_rows, test_cols), data)
    metric_value = conditional.metric_observed(loss_name, data_fold['test'])
    return {'loss': metric_value}


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
        _ops.cv_fit_eval(spec, data, fold_count, loss_name, seed)
        for spec in specs
        ]

    return list(zip(specs, results))

# Pure meta-ops
# =============

def cv_fit_eval(model_spec, data, fold_count=10, loss_name='RSS', seed=0, _ops=_self):
    """
    Fits and eval a model on each fold of the data and return the total loss

    Returns:
        a dict with keys:
            'folds': list of per-fold results, each a dict with keys:
                'loss': the loss value
            'loss': the total loss
    """
    folds = []
    for i in range(fold_count):
        train, test, _ = _ops.fold(data, i, fold_count, seed=seed)
        folds.append(
            _ops.fit_eval(model_spec, {'train': train, 'test': test}, loss_name)
            )
    total_loss = _ops.aggregate_losses([_fold['loss'] for _fold in folds])

    return {'folds': folds, 'loss': total_loss}
