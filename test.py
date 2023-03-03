"""
Misc tests for matrix-t dists
"""

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from block import JaxBlockMatrix
import block_mt as bmt

mkb = JaxBlockMatrix.from_dense

def test_block_from_scalar() -> None:
    """
    Create trivial block from a scalar and get it back
    """
    assert mkb(1) == 1

def test_block_from_blocks() -> None:
    """
    Create block for various dimensions of blocks
    """
    blk = JaxBlockMatrix.from_blocks({
        (0, 0): 1, # scalar
        (0, 1): np.array([2, 3]), # vector bcasting as row vector
        (1, 0): np.array([[4, 5]]).T, # explicit column vector
        (1, 1): np.array([[6, 7], [8, 9]]) # matrix
        })

    assert_equal(blk.to_dense(), np.array([
        [1, 2, 3],
        [4, 6, 7],
        [5, 8, 9]
        ]))

def test_block_from_blocks_nested() -> None:
    """
    Create block for various dimensions of blocks, including block objects as
    blcoks
    """

    blk = JaxBlockMatrix.from_blocks({
        (0, 0): mkb(1), # scalar
        (0, 1): mkb(np.array([2, 3])), # vector bcasting as row vector
        (1, 0): mkb(np.array([[4, 5]]).T), # explicit column vector
        (1, 1): mkb(np.array([[6, 7], [8, 9]])) # matrix
        })

    assert_equal(blk.to_dense(), np.array([
        [1, 2, 3],
        [4, 6, 7],
        [5, 8, 9]
        ]))

@pytest.mark.skip
def test_batch_ncmt() -> None:
    """
    Compute ncmt log pdfs for a batch of values at once
    """
    observed = np.array([
        [0., 1.],
        [3., 2.],
        [-1., 1.]
        ])[..., None, :]

    left = np.eye(1)
    right = np.eye(2)
    dfs = 1.

    dist = bmt.NonCentralMatrixT.from_params(dfs, left, right)

    iterative_pdfs = np.stack([
        dist.log_pdf(x)
        for x in observed
        ])
    batched_pdfs = dist.log_pdf(observed)

    assert_allclose(iterative_pdfs, batched_pdfs)
