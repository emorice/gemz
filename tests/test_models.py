"""
Model tests.

Limited to check that all defined models can complete a fit and predict, but
without guarantees of the correctness of the predictions
"""

import pytest

import numpy as np

import gemz

@pytest.fixture
def data():
    """
    Uncorrelated data with no signal whatsoever
    """
    return np.random.default_rng(0).normal(size=(2, 10, 20))

model_specs = [
    {
        'model': 'linear'
        },
    {
        'model': 'kmeans',
        'n_clusters': 2
        },
    {
        'model': 'wishart'
        },
    {
        'model': 'linear_shrinkage'
        },
    {
        'model': 'linear_shrinkage_cv'
        },
    {
        'model': 'lscv_precision_target'
        },
    {
        'model': 'lscv_free_diagonal'
        },
    {
        'model': 'nonlinear_shrinkage'
        },
    {
        'model': 'cmk'
        },
    {
        'model': 'gmm'
        },
    {
        'model': 'igmm'
        },
    ]

@pytest.mark.parametrize('model_spec', model_specs)
def test_fit_predict_null(data, model_spec):
    """
    Fit and predict from the fitted model
    """

    train, test = data

    fit = gemz.models.fit(model_spec, train)

    predictions = gemz.models.predict_loo(model_spec, fit, test)

    assert predictions.shape == test.shape
