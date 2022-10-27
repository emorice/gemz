"""
Tests the linear algebra utils
"""

import pytest

import numpy as np

from gemz import linalg
from gemz.models import lscv_loo

def assert_dets_regular(matrix, other):
    s_matrix, l_matrix = np.linalg.slogdet(matrix)
    s_other, l_other = np.linalg.slogdet(other)

    assert s_matrix == s_other
    assert np.allclose(l_matrix, l_other)

@pytest.mark.parametrize('weight', [2., 0.])
def test_lru_inv(weight):
    """
    Test LRU matrix inversion
    """

    matrix = np.arange(9).reshape(3, 3) + np.eye(3)
    vec_left = np.arange(3).reshape(3, 1)
    vec_right = np.arange(3, 6).reshape(1, 3)

    lru = linalg.LowRankUpdate(
        matrix, vec_left, weight, vec_right
        )

    inv_lru = np.linalg.inv(lru)

    concrete_lru = matrix + weight * vec_left @ vec_right

    assert np.allclose(inv_lru @ concrete_lru, np.eye(3))
    assert np.allclose(concrete_lru @ inv_lru, np.eye(3))

    assert_dets_regular(concrete_lru, lru)

def test_loo_residuals():
    """
    Test LOO linear predictions
    """

    data = np.array([1.] * 5 + [-1.] * 5)[:, None] * np.ones(19)
    data += np.random.default_rng(0).normal(size=data.shape)

    reg = 1e-4 # Empirical threshold under which mild unstability appears
    # Catastrophic unstability starts around 1e-7

    ref_residuals = np.empty_like(data)

    for i, row in enumerate(data):
        loo_data = data[np.arange(len(data)) != i]
        loo_cov = reg * np.eye(data.shape[1]) + loo_data.T @ loo_data / len(loo_data)
        loo_prec = np.linalg.inv(loo_cov)

        ref_residuals[i] = (loo_prec @ row) / np.diagonal(loo_prec)

    test_residuals, test_precisions = lscv_loo._loo_predict(data, reg)

    assert np.allclose(ref_residuals, test_residuals)
