"""
Misc tests for matrix-t dists
"""

import warnings

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from gemz.jax.linalg.block import JaxBlockMatrix
import gemz.stats.matrixt as bmt
from gemz.linalg import Identity

mkb = JaxBlockMatrix.from_dense

@pytest.fixture(autouse=True)
def no_deprecated_code():
    """
    Pytest fixture to treat deprecation warnings as errors
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=DeprecationWarning)
        yield

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
    blocks
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

def test_block_inv() -> None:
    """
    Invert a block matrix
    """
    matrix = np.array([
        [4, 8, 6, 7],
        [7, 9, 3, 0],
        [2, 7, 6, 3],
        [7, 9, 8, 0]
        ])

    assert_allclose(
        np.linalg.inv(mkb(matrix)).to_dense(),
        np.linalg.inv(matrix),
        rtol=5e-6,
    )

def test_block_swapaxes() -> None:
    """
    Transpose a block matrix
    """
    mat = JaxBlockMatrix.from_blocks(
            { (0, 0, 0): np.zeros((2, 3, 4))}
            )
    assert mat.dims == ({0: 2}, {0: 3}, {0: 4})
    assert mat[0, 0, 0].shape == (2, 3, 4)

    mat_t = np.swapaxes(mat, -1, -2)
    assert mat_t.dims == ({0: 2}, {0: 4}, {0: 3})
    assert mat_t[0, 0, 0].shape == (2, 4, 3)

def test_block_ucast() -> None:
    """
    Upcast a matrix before summing with an other
    """
    left = mkb(np.ones((1, 3, 4)))
    right = mkb(np.ones((3, 4)))

    res = left + right

    assert_allclose(res.to_dense(), np.ones((1, 3, 4))*2)

def test_block_bcast() -> None:
    """
    Broadcast a vector before summing with an other
    """
    left = mkb(np.ones((1,))) # One block
    right = mkb(np.ones((3,))) # Three blocks

    res = left + right

    assert_allclose(res.to_dense(), np.ones((3,))*2)

def test_block_solve_batched() -> None:
    """
    Test a batched solve
    """
    linmap = JaxBlockMatrix.from_blocks({ tuple(): 2.*np.ones((3, 1, 1)) })
    np_targets = np.arange(6).reshape(3, 1, 2)
    targets = JaxBlockMatrix.from_blocks({ tuple(): np_targets })

    assert_allclose(
            np.linalg.solve(linmap, targets).to_dense(),
            .5 * np_targets
            )

def test_block_solve_upcast() -> None:
    """
    Test an upcast (adding dimensions) batched solve
    """
    linmap = JaxBlockMatrix.from_blocks({ tuple(): 2.*np.ones((1, 1)) })
    np_targets = np.arange(6).reshape(3, 1, 2)
    targets = JaxBlockMatrix.from_blocks({ tuple(): np_targets })

    assert_allclose(
            np.linalg.solve(linmap, targets).to_dense(),
            .5 * np_targets
            )

def test_block_index_ellipsis() -> None:
    """
    Index a ND block with ellipses
    """
    np_array = np.arange(30).reshape((2, 3, 5))
    array = mkb(np_array)

    # Note that selecting the "block" 4 means selecting a singleton of columns,
    # not a single column. This is a point where indexing semantics differ
    # between block and pure arrays.
    np_sub = np_array[..., [4]]
    sub = array[..., 4]

    assert_allclose(sub.to_dense(), np_sub)

def test_block_identity() -> None:
    """
    Build a block matrix out of virtual identity matrices
    """
    mat = JaxBlockMatrix.from_blocks({(0, 0): Identity(2), (1, 1): Identity(1)})

    assert_allclose(np.asarray(mat), np.eye(3))

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
        dist.logpdf(x)
        for x in observed
        ])
    batched_pdfs = dist.logpdf(observed)

    assert_allclose(iterative_pdfs, batched_pdfs)

def test_batch_ncmt_uni_cond() -> None:
    """
    Compute ncmt uni conditional for a batch of values at once
    """
    observed = np.array([
        [0., 1., 2.],
        [3., 2., -1.],
        ])[..., None, :]

    left = np.eye(1)
    right = np.eye(3)
    dfs = 1.

    dist = bmt.NonCentralMatrixT.from_params(dfs, left, right, gram_mean_left=1.)

    iterative_ucs = [np.stack(stat) for stat in zip(*(
        dist.uni_cond(x)
        for x in observed
        ))]
    batched_ucs = dist.uni_cond(observed)

    for istat, bstat in zip(iterative_ucs, batched_ucs):
        # Why isn't this closer ? The sequence of floating point operations
        # should be essentially identical
        assert_allclose(bstat, istat, rtol=5e-6)
