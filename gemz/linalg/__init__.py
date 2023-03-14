"""
Linear algebra utils
"""

import numpy as np

from .virtual_array import ImplicitMatrix

from .basics import Identity, ScaledIdentity, Diagonal, ScaledMatrix

class LowRankUpdate(ImplicitMatrix):
    """
    Symmetric low-rank update of a matrix, stored symbolically.

    This allow efficient storage and operations such as inversion, determinant
    or matrix-vector product (if applicable) when:
        * the update has a noticeably smaller rank that the dimension of the
          matrix, and
        * storage and operations of the updatee can be done at a small marginal
          cost, because it has a simple structure (e.g. a diagonal matrix) or is
          broadcasted against many different updates.

    This representation is not identifiable and not meant to be.
    """
    def __init__(self, base, factor_left, weight, factor_right=None):
        """
        Builds a symbolic representation of the matrix
            base + factor_left @ weight @ factor_right

        All inputs can have (matching) leading batch dimensions.
        If base or weight have less than two dimensions, there are assumed to be
        diagonal matrices or scalar multiple of the identity matrix.

        Note that you can use a negative weight to write a downdate.

        Args:
            factor_right: if not given, defaults to the batched transpose of
                factor_left, yielding a symmetric low rank update.
        """
        self.base = as_matrix(base, factor_left.shape[-2])
        self.factor_left = factor_left
        self.weight = as_matrix(weight, factor_left.shape[-1])

        if factor_right is None:
            self.factor_right = np.swapaxes(factor_left, -2, -1)
        else:
            self.factor_right = factor_right

        self._inv_base = None
        self._inv_weight = None
        self._capacitance = None

    class LRUShape:
        """
        Shape proxy for LRU matrix

        The whole shape is not easily predicted but specific lengths may be
        guessed easily.
        """
        def __init__(self, lru):
            self.lru = lru

        def __getitem__(self, index):
            if index == -1:
                return max(
                    self.lru.base.shape[-1],
                    self.lru.factor_right.shape[-2],
                    )
            raise NotImplementedError(index)

    @property
    def shape(self):
        """
        Shape accessor object
        """
        return self.LRUShape(self)

    def __matmul__(self, right):
        """
        Matmul of `self @ right`

        Nothing special about the implementation except being careful with the
        associativity
        """
        return (
            self.base @ right
            + self.factor_left @ (
                self.weight @ (
                    self.factor_right  @ right
                    )
                )
            )

    def __rmatmul__(self, left):
        """
        Matmul of `left @ self`

        Nothing special about the implementation except being careful with the
        associativity
        """
        return (
            left @ self.base
            + (
                (left @ self.factor_left) @ self.weight
                ) @ self.factor_right
            )

    def diagonal(self):
        """
        Concrete diagonal of the matrix
        """
        return (
            np.diagonal(self.base)
            + np.sum(
                (self.factor_left @ self.weight) # Bni @ Bij -> Bnj
                * np.swapaxes(self.factor_right, -2, -1), # Bjm -> Bmj, m = n
                -1) # Bni -> Bn
            )

    @property
    def inv_base(self):
        """
        Inverse of base matrix, cached
        """
        if self._inv_base is None:
            self._inv_base = np.linalg.inv(self.base)
        return self._inv_base

    @property
    def capacitance(self):
        """
        Capacitance matrix, cached
        """
        if self._capacitance is None:
            inner_dim = max(
                self.factor_right.shape[-2],
                self.weight.shape[-1]
                )
            self._capacitance = (
                np.eye(inner_dim)
                + (self.factor_right @
                    (self.inv_base @ self.factor_left)
                  ) @ self.weight
                )
        return self._capacitance

    def inv(self):
        """"
        Representation of inverse through Woodbury identity
        """
        return LowRankUpdate(
            base=self.inv_base,
            factor_left=self.inv_base @ self.factor_left,
            weight=-(self.weight @ np.linalg.inv(self.capacitance)),
            factor_right=self.factor_right @ self.inv_base,
            )

    def slogdet(self):
        """
        Sign and log-determinant through matrix determninant lemma
        """
        s_base, l_base = np.linalg.slogdet(self.base)
        s_capa, l_capa = np.linalg.slogdet(self.capacitance)

        return s_base * s_capa, l_base + l_capa

    def __truediv__(self, divisor):
        """
        Self / divisor

        Divisor must be a batched scalar, and must be manually broadcasted if
        needed first.

        Scalar is required as Hadamard division of LRU is not easily doable.

        Automatic broadcasting of scalars or (constant) vectors is not
        implemented as the main use case is batched scalars that require
        pre-broadcasting anyway.
        """
        if divisor.shape[-2:] != (1, 1):
            raise ValueError('Cannot divide a LowRankUpdate by something else '
                'than a batched 1 x 1 array. Received divisor with shape ' +
                str(divisor.shape))
        return ScaledMatrix(self, 1. / divisor[..., 0, 0])

class SymmetricLowRankUpdate(LowRankUpdate):
    """
    Symmetric low-rank update of a matrix, stored symbolically.

    Special case of LowRankUpdate,
    """
    def __init__(self, base, factor, weight):
        """
        Builds a symbolic representation of the matrix
            base + factor @ weight @ factor.T

        All inputs can have (matching) leading batch dimensions.
        If base or weight have less than two dimensions, there are assumed to be
        diagonal matrices or scalar multiple of the identity matrix.

        Note that you can use a negative weight to write a downdate.
        """
        super().__init__(base, factor, weight)

    def inv(self):
        """"
        Representation of inverse through Woodbury identity

        Preserve the symmetry, hence avoiding duplicating the factor as
        LowRankUpdate would do.
        """
        return SymmetricLowRankUpdate(
            base=self.inv_base,
            factor=self.inv_base @ self.factor_left,
            weight=-(self.weight @ np.linalg.inv(self.capacitance))
            )

def as_matrix(obj, bcast_inner_dim):
    """
    Performs conversion of scalars into implicit multiples of the identity matrix, and
    later 1-d arrays into implicit diagonal arrays.

    The dimension of the potential identity matrix to use must be specified
    """
    # Implicit objects may not have a concrete shape, but they already matrices
    if isinstance(obj, ImplicitMatrix):
        return obj

    # This should work for a wide range of non-numpy objects
    shape = np.shape(obj)

    if len(shape) == 0:
        return ScaledIdentity(obj, inner_dim=bcast_inner_dim)
    if len(shape) == 1:
        return Diagonal(obj)
    return obj

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
