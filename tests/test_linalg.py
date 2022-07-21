"""
Tests the linear algebra utils
"""

import pytest

import numpy as np

from gemz import linalg

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
