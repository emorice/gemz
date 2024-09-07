"""
Model tests.

Limited to check that all defined models can complete a fit and predict, but
without guarantees of the correctness of the predictions
"""

import pytest

import numpy as np

import gemz
from gemz.model import Model, EachIndex

# pylint: disable=redefined-outer-name

@pytest.fixture
def unsplit_data():
    """
    Rank one matrix with some added noise
    """
    # 11 obs 20 vars train, 15 obs 20 vars test
    shape = (11 + 15, 20)

    return (
        np.ones(shape) +
        np.random.default_rng(0).normal(size=shape)
        )

@pytest.fixture
def data(unsplit_data):
    """
    Same data but already split into two train and test arrays along axis 0
    """
    return np.split(unsplit_data, [11])

model_specs = [
    { 'model': 'linear' },
    { 'model': 'kmeans', 'n_groups': 2 },
    { 'model': 'wishart' },
    { 'model': 'linear_shrinkage', 'prior_var': 1. },
    { 'model': 'lscv_loo'},
    { 'model': 'lscv_precision_target' },
    { 'model': 'lscv_free_diagonal', 'scale': None },
    { 'model': 'lscv_free_diagonal', 'scale': 1. },
    { 'model': 'nonlinear_shrinkage' },
    { 'model': 'cmk', 'n_groups': 2 },
    { 'model': 'gmm', 'n_groups': 2 },
    { 'model': 'igmm', 'n_groups': 2 },
    { 'model': 'svd', 'n_factors': 2 },
    { 'model': 'svd', 'n_factors': 2, 'revision': 2 },
    { 'model': 'peer', 'n_factors': 2 },
    { 'model': 'peer', 'n_factors': 2, 'reestimate_precision': True },
    { 'model': 'mt_sym'},
    { 'model': 'cv', 'inner': {'model': 'linear_shrinkage' } },
    { 'model': 'cv', 'inner': {'model': 'svd'} },
    { 'model': 'cv', 'inner': {'model': 'svd', 'revision': 2} },
    { 'model': 'cv', 'inner': {'model': 'svd'}, 'grid_max': 3},
    { 'model': 'cv', 'inner': {'model': 'peer'}, 'fold_count': 3, 'grid_size': 3},
    { 'model': 'cv', 'inner': {'model': 'cmk'}, 'fold_count': 3, 'grid_size': 3},
    # Can't test 2-d cv with less than, say, 2 x 2
    { 'model': 'cv', 'inner': {'model': 'gmm'}, 'fold_count': 3, 'grid_size': 4},
    { 'model': 'cv', 'inner': {'model': 'igmm'}, 'fold_count': 3, 'grid_size': 3},
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

# Some specs are shared but the models have fundamentally different interfaces
model_specs_v2 = [
    { 'model': 'linear' },
    { 'model': 'mt_std' },
    # Only makes sense with fixed parameters, not supported yet
    #{ 'model': 'mt_sym', 'scale': 10. },
    ]
@pytest.mark.parametrize('model_spec', model_specs_v2, ids=model_id)
def test_block_loo(unsplit_data, model_spec):
    """
    Block-loo conditional matches naive iterated block-block
    """
    data = unsplit_data
    len0, len1 = np.shape(data)

    len_obs0 = len0 // 2

    model = gemz.model.Model.from_spec(model_spec)

    block_means = np.zeros((len0 - len_obs0, len1))

    for jcol in range(len1):
        block_means[:, [jcol]] = model.conditional[len_obs0:, jcol](data).mean

    loo_means = model.conditional[len_obs0:, EachIndex](data).mean

    np.testing.assert_allclose(block_means, loo_means)
    for stat in ['min', 'median', 'mean', 'max']:
        print('abs', stat, getattr(np, stat)(np.abs(block_means - loo_means)))
        print('rel', stat, getattr(np, stat)(np.abs(block_means - loo_means) * 2. / np.abs(block_means + loo_means)))
