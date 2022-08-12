"""
Unified model interface
"""

import sys

from . import cv

_METHODS = {}

def get(name):
    """
    Returns a model by name
    """
    return _METHODS[name]

def add(name):
    """
    Register a model class by name
    """
    def _set(cls):
        _METHODS[name] = cls
        return cls
    return _set

def add_module(name, module_name):
    """
    Register a model module by name
    """
    _METHODS[name] = sys.modules[module_name]

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

def eval_loss(model_spec, model_fit, test_data, loss_name):
    """
    Simple wrapper around the losses in `cv`.

    Factored out as its own function to make it easy to pipeline
    """
    loss_fn = cv.LOSSES[loss_name]

    model = get(model_spec['model'])

    return loss_fn(model, model_fit, test_data)
