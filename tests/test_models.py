"""
Model tests.

Limited to check that all defined models can complete a fit and predict, but
without guarantees of the correctness of the predictions
"""

import pytest

import numpy as np

import gemz

# pylint: disable=redefined-outer-name

@pytest.fixture
def data():
    """
    Uncorrelated data with no signal whatsoever
    """
    # 11 obs 20 vars train, 15 obs 20 vars test
    return np.split(
        np.random.default_rng(0).normal(size=(26, 20)),
        [11])

model_specs = [
    { 'model': 'linear' },
    { 'model': 'kmeans', 'n_groups': 2 },
    { 'model': 'wishart' },
    { 'model': 'linear_shrinkage', 'prior_var': 1. },
    { 'model': 'cv', 'inner': {'model': 'linear_shrinkage' } },
    { 'model': 'lscv_precision_target' },
    { 'model': 'lscv_free_diagonal', 'scale': None },
    { 'model': 'lscv_free_diagonal', 'scale': 1. },
    { 'model': 'nonlinear_shrinkage' },
    { 'model': 'cmk', 'n_groups': 2 },
    { 'model': 'gmm', 'n_groups': 2 },
    { 'model': 'igmm', 'n_groups': 2 },
    { 'model': 'svd', 'n_factors': 2 },
    { 'model': 'peer', 'n_factors': 2 },
    { 'model': 'peer', 'n_factors': 2, 'reestimate_precision': True },
    { 'model': 'cv', 'inner': {'model': 'svd'} }
    ]

def model_id(model):
    """
    Printable model name for tests
    """
    return gemz.models.get_name(model)

@pytest.mark.parametrize('model_spec', model_specs, ids=model_id)
def test_fit_predict_null(data, model_spec):
    """
    Fit and predict from the fitted model
    """

    train, test = data

    fit = gemz.models.fit(model_spec, train)

    predictions = gemz.models.predict_loo(model_spec, fit, test)

    assert predictions.shape == test.shape
