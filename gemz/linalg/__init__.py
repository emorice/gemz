"""
Linear algebra utils
"""

import numpy as np

from .virtual_array import ImplicitMatrix

from .basics import Identity, ScaledIdentity, Diagonal, ScaledMatrix
from .low_rank import LowRankUpdate, SymmetricLowRankUpdate

def loo_sum(array, axis=None):
    """
    Concrete leave-one-out sum over an array.

    Note that for a sum the memory used by the result is the same as for the input.
    """
    return np.sum(array, axis=axis, keepdims=True) - array

def loo_matmul(left, right):
    """
    Concrete leave-one-out matrix multiplication over an array.

    The LOO axis is left in the middle, signature is (a, b), (b, c) -> (a, b, c)
    for 2D inputs.

    Note that in contrast with loo_sum, a concrete loo matmul may be much larger that
    its inputs. Consider using implicit matrices.
    """
    joint = left[..., None] * right
    return loo_sum(joint, -2)

def loo_square(array_bnp, weights_bp, reg_b=np.array(0.)):
    """
    Implicit leave-one-out transpose square of an array.

    With weights along the contracted dimension. [Loo-]contracts over the last
    dimension of `array`, like np.cov or np.corrcoef. Puts the LOO axis before the
    duplicated axis. Supports leading batching dimensions.
    Signature (..ij, ..j) -> (..jii)
    """

    # Batched transpose
    array_bpn = np.swapaxes(array_bnp, -2, -1)
    # Non-loo square
    base_bnn = (array_bnp * weights_bp[..., None, :]) @ array_bpn

    base_bnn += ScaledIdentity(reg_b, array_bnp.shape[-2])

    return SymmetricLowRankUpdate(
        # Insert a broadcasting loo dimension just before the duplicated one
        base=base_bnn[..., None, :, :],
        # Add a dummy contracting dimension at the end
        factor=array_bpn[..., None],
        # Add two dummy contracting dimensions, and
        # swap the sign since LOO is a *down*date
        weight=-(weights_bp[..., None, None])
        )

def loo_cross_square(left_bnp, weights_bp, right_bmp):
    """
    Like `loo_square` with possibly different left and right factors. So
    essentially a contraction like matmul, but with indexing conventions
    matching loo_square, not loo_matmul

    Signature (..ij, ..j, ..kj) -> (..jik)
    """

    right_bpm = np.swapaxes(right_bmp, -2, -1)
    left_bpn = np.swapaxes(left_bnp, -2, -1)

    # Non-loo cross square
    base_bnm = (left_bnp * weights_bp[..., None, :]) @ right_bpm

    return LowRankUpdate(
        # Insert a broadcasting loo dimension just before the outer'ed ones
        base=base_bnm[..., None, :, :],
        # Put the trailing LOO axis at the end of the batching dimensions,
        # then add a dummy contracting dimension at the end (bpn1)
        factor_left=left_bpn[..., None],
        # Add two dummy contracting dimensions, and
        # swap the sign since LOO is a *down*date
        weight=-weights_bp[..., None, None],
        # Add dummy contraction dim as second to last (bp1m)
        factor_right=right_bpm[..., None, :]
        )

def eager_map(function, *arrays):
    """
    Convenience function to map a function over several arrays, calling
    accelerated implementation if available, else falling back to a python loop.

    To keep simple compatibility with numpy, this has several differences with
    the more generic jax.lax.map.
     * arrays must be a tuple of arrays, no other pytree possible
     * function must take positional array arguments (one array per argument)
     * function must return a tuple of arrays, no other pytree possible. For
       eager differentiation, it must more precisely return a tuple of scalar
       arrays.
    """
    if arrays and hasattr(arrays[0], 'eager_map'):
        return arrays[0].eager_map(function, *arrays)
    return tuple(
        np.stack(ret) for ret in
        zip(*[function(*args) for args in zip(*arrays)])
        )
