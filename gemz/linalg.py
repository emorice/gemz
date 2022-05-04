"""
Linear algebra utils
"""

import numpy as np

class ImplicitMatrix:
    """
    Base class for integration of implicit matrices with regular numpy code.

    Handle dispatching through numpy protocols
    """

    def __array_ufunc__(self, ufunc, method, *args):
        if ufunc == np.matmul and method == '__call__':
            return self.matmul(*args)
        return NotImplemented

    def __array_function__(self, func, _, args, kwargs):
        if func is np.diagonal:
            if len(args) != 1 or kwargs or args[0] is not self:
                return NotImplemented
            if hasattr(self, 'diagonal'):
                return self.diagonal()
        return NotImplemented

    @classmethod
    def matmul(cls, *args):
        """
        Limited implememtation of numpy.matmul
        """
        if len(args) != 2:
            return NotImplemented

        left, right = args

        if isinstance(left, cls):
            return left.matmul_right(right)

        if isinstance(right, cls):
            return right.matmul_left(left)

        return NotImplemented


class ScaledIdentity(ImplicitMatrix):
    """
    Implicit scalar multiple of the identity matrix
    """
    def __init__(self, scalar):
        self.scalar = scalar

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

    def diagonal(self):
        """
        Diagonal as a scalar. The result has a undefined shape, we rely on
        broadcasting in the context where `diagonal` is called.
        """
        return self.scalar

class RWSS(ImplicitMatrix):
    """
    Regularized weighted symmetric square of a matrix, stored symbolically.

    This allow efficient storage and operations such as inversion and
    determinant when the squared matrix is much taller than wide and thus
    smaller than its dense square.

    This representation is not identifiable and not meant to be.
    """

    def __init__(self, base, factor, weight):
        """
        Builds a symbolic represenation of the matrix
            base + factor @ weight @ factor.T
        """
        self.base = as_matrix(base)
        self.factor = factor
        self.weight = as_matrix(weight)


    def matmul_right(self, right):
        """
        Matmul of `self @ right`

        Nothing special about the implementation except being careful with the
        associativity
        """
        return (
            self.base @ right
            + self.factor @ (
                self.weight @ (
                    self.factor.T @ right
                    )
                )
            )

    def diagonal(self):
        """
        Concrete diagonal of the matrix
        """
        return (
            np.diagonal(self.base)
            + np.sum((self.factor @ self.weight) * self.factor, 1)
            )

def as_matrix(obj):
    """
    Performs conversion of scalars into implicit multiples of the identity matrix, and
    later 1-d arrays into implicit diagonal arrays.
    """
    # This should work for a wide range of non-numpy objects
    shape = np.shape(obj)

    if len(shape) == 0:
        return ScaledIdentity(obj)
    if len(shape) == 1:
        raise NotImplementedError
    if len(shape) == 2:
        return obj

    raise NotImplementedError
