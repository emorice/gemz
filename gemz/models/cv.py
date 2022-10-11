"""
Cross-validation utils
"""

import numpy as np

from . import methods, ops

methods.add_module('cv', __name__)

LOSSES = {}

def loss(name):
    """
    Record a named loss function
    """
    def _wrap(function):
        LOSSES[name] = function
        return function
    return _wrap

@loss('RSS')
def rss_loss(method, model, test):
    """
    Evaluate a fitted mode with a classical RSS loss
    """

    predictions = method.predict_loo(model, test)
    return np.sum((test - predictions)**2)

@loss('iRSS')
def irss_loss(method, model, test):
    """
    Classical RSS loss except not aggregated over the working dimensions

    (so only summed over the replicates in the fold)
    """

    predictions = method.predict_loo(model, test)
    return np.sum((test - predictions)**2, 0)

@loss('NAIC')
def naic_loss(method, model, test):
    """
    Negative Akaike Information Criterion loss

    Untested.
    """
    log_pdfs = method.log_pdf(model, test)
    return - np.sum(log_pdfs)

@loss('GEOM')
def geom_loss(method, model, test):
    """
    Geometric aggregate of RSS over dimensions.

    More robust than RSS with heteroskedastic data.
    """
    predictions = method.predict_loo(model, test)
    dim_squares = np.sum((test - predictions)**2, -1)

    return np.sum(np.log(dim_squares))

def fit(data, inner, fold_count=10, seed=0, loss_name="RSS", grid_size=20,
        grid=None, _ops=ops):
    """
    Fit and eval the given method on folds of data

    Args:
        data: N1 x N2. Models are fixed-dimension N2 x N2, and cross-validation
            is performed along N1.
        grid: if given, use this explicitely defined grid instead of generating
        one.
    """

    specs = _ops.build_eval_grid(
                inner, data, fold_count, loss_name,
                grid_size, grid,
                seed, _ops=_ops
            )

    best_model = _ops.select_best(specs)

    return {
        'inner': inner,
        'loss_name': loss_name,
        'selected': best_model,
        'fit': _ops.fit(best_model, data),
        'grid': specs,
        }

def predict_loo(model_fit, new_data):
    """
    Linear shrinkage loo prediction for the best model found during cv.
    """
    inner_model = methods.get(model_fit['inner']['model'])
    inner_fit = model_fit['fit']
    return inner_model.predict_loo(inner_fit, new_data)

def get_name(spec):
    """
    Readable description
    """
    return f"{spec['model']}/{spec['inner']['model']}"
