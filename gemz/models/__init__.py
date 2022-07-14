"""
Module containing all statistical model fit and prediction code
"""
import sys

# TODO: this should have some discovery scheme
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

def get(name):
    """
    Returns a model by name
    """
    return getattr(sys.modules[__name__], name)

def fit(model_spec, train_data):
    """
    Fit a model from a model specification.

    Args:
        model_spec: a dictionnary containing the name of the model in 'model',
            and keyword arguments to pass along to the fit function of
            said model
        train_data: data to pass to fit
    """

    model = get(model_spec['model'])
    kwargs = dict(model_spec)
    del kwargs['model']

    return model.fit(train_data, **kwargs)

def predict_loo(model_spec, model_fit, test_data):
    """
    Like `fit` for the `predict_loo` method
    """
    return get(model_spec['model']).predict_loo(model_fit, test_data)
