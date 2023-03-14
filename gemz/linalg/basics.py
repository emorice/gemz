"""
Elementary virtual arrays like identity, diagonals, and scalar multiples
"""

import dataclasses as dc
from typing import Any

import numpy as np

from .virtual_array import ImplicitMatrix

@dc.dataclass
class ScaledIdentity(ImplicitMatrix):
    """
    Implicit scalar multiple of the identity matrix
    """
    scalar: Any
    inner_dim: int

    @property
    def shape(self):
        """
        Array shape
        """
        return np.shape(self.scalar) + (self.inner_dim, self.inner_dim)

    def matmul_right(self, right):
        """
        Matmul of self @ right
        """
        return self.scalar * right

    def matmul_left(self, left):
        """
        Matmul of left @ self
        """
        return left * self.scalar

    def __sub__(self, right):
        """
        self - right

        Fall back to dense array
        """
        return self._as_dense() - right

    def diagonal(self):
        """
        Diagonal as a scalar. The result has a undefined shape, we rely on
        broadcasting in the context where `diagonal` is called.
        """
        return self.scalar

    def inv(self):
        """
        Inverse scalar times identity, raises if 0
        """
        return self.__class__(
            scalar=1.0 / self.scalar,
            inner_dim=self.inner_dim)

    def _solve(self, rhs):
        return rhs / self.scalar

    def add(self, other, out=None):
        """
        Concrete addition
        """
        # For now we only consider adding to a concrete array, in which case we
        # can concretize self
        return np.add(
            other,
            self.scalar[..., None, None] * np.eye(self.inner_dim),
            out=out)

    def slogdet(self):
        """
        Sign and log det of self
        """
        return (
            np.sign(self.scalar) ** self.inner_dim,
            self.inner_dim * np.log(np.abs(self.scalar))
            )

    def _broadcast_to(self, shape):
        """
        Broadcast self to shape
        """
        if len(shape) < 2:
            raise TypeError(f'Cannot broadcast matrix to {shape}')
        if shape[-1] != shape[-2]:
            raise TypeError(f'Shape must be square: {shape}')
        if self.inner_dim not in (shape[-1], 1):
            raise TypeError(f'Cannot broadcast matrix from {self.shape} to {shape}')

        scalar = self.broadcast_to(self.scalar, shape[:-2])
        return self.__class__(scalar, shape[-1])

    def _as_dense(self):
        return (
            self.broadcast_to(self.scalar, self.shape) *
            self.aa.eye(self.inner_dim)
            )

class ScaledMatrix(ImplicitMatrix):
    """
    Implicit scalar multiple of an arbitrary matrix.

    Note that the scalar and matrix may be batched, so "scalar" and "matrix"
    means any tensors with D and D+2 dimensions really.

    This is useful when the base matrix is broadcasted along an axis with
    different scalars (effectively an outer product between a matrix and a
    vector). This include cases where the base is itself an ImplicitMatrix that
    wraps a broadcasted matrix, though the scalar could arguably be pushed down
    in that case.
    """
    def __init__(self, base, multiplier):
        self.base = base
        self.multiplier = multiplier

    def inv(self):
        """
        Possibly batched inverse.
        """
        return ScaledMatrix(
            base=np.linalg.inv(self.base),
            multiplier=1. / self.multiplier
            )

    def matmul_right(self, right):
        """
        Self @ right

        By default we multiply with the base first then apply the multiplier, so
        that with a broadcasted base and right we save the expansion for the
        end.
        """
        return self.multiplier[..., None, None] * (self.base @ right)

    def matmul_left(self, left):
        """
        Left @ self

        By default we multiply with the base first then apply the multiplier, so
        that with a broadcasted base and left we save the expansion for the
        end.
        """
        return self.multiplier[..., None, None] * (left @ self.base)

    def slogdet(self):
        """
        Sign and log det of self
        """
        s_base, l_base = np.linalg.slogdet(self.base)
        dim = self.base.shape[-1]
        return (
            s_base * np.sign(self.multiplier) ** dim,
            l_base + dim * np.log(np.abs(self.multiplier))
            )

def Identity(inner_dim):
    return ScaledIdentity(1.0, inner_dim)

class Diagonal(ImplicitMatrix):
    """
    Implicit diagonal matrix

    The diagonal can be a batch of diagonal vectors, in which case this class
    behaves like a batch of of diagonal matrices
    """
    def __init__(self, diagonal):
        self._diagonal = diagonal

    @property
    def shape(self):
        # Duplicated last dim
        return self._diagonal.shape + (self._diagonal.shape[-1],)

    def diagonal(self):
        """
        Diagonal as a vector
        """
        return self._diagonal

    def inv(self):
        """
        Diagonal inverse of diagonal matrix
        """
        return Diagonal(1. / self._diagonal)

    def matmul_right(self, right):
        """
        Matmul of self @ right, i.e. multiplying elementwise each column
        """
        return self._diagonal[..., :, None] * right

    def matmul_left(self, left):
        """
        Matmul of left @ self, i.e. multiplying elementwise each row
        """
        return left * self._diagonal[..., None, :]

    def _broadcast_to(self, shape):
        """
        Broadcast implementation
        """
        if len(shape) < 2 or shape[-1] != shape[-2]:
            raise TypeError(f'Cannot broadcast to {shape}')

        new_diag_shape = shape[:-1]
        new_diag = self.broadcast_to(self._diagonal, new_diag_shape)
        return self.__class__(new_diag)

    def _as_dense(self):
        return self._diagonal * self.aa.eye(self.shape[-1])

    def __sub__(self, right):
        """
        Fallback to dense to compute self - right
        """
        return self._as_dense() - right
