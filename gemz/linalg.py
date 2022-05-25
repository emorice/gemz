"""
Linear algebra utils
"""

import numpy as np

# The __array_* protocol methods actually define a public interface but pylint
# does not know that
# pylint: disable=too-few-public-methods
class ImplicitMatrix:
    """
    Base class for integration of implicit matrices with regular numpy code.

    Handle dispatching through numpy protocols
    """
    _implementations = {}

    @classmethod
    def implements(cls, func):
        """
        Function decorator to register an implementation of a numpy function
        """
        def _wrapper(func_imp):
            cls._implementations[func] = func_imp
            return func_imp
        return _wrapper

    def __array_ufunc__(self, ufunc, method, *args):
        if ufunc in self._implementations and method == '__call__':
            imp = self._implementations[ufunc]
            return imp(self, *args)
        return NotImplemented

    def __array_function__(self, func, _, args, kwargs):
        if func in self._implementations:
            # Only unary functions are handled for now
            if len(args) != 1 or kwargs or args[0] is not self:
                return NotImplemented
            imp = self._implementations[func]
            return imp(self, args)
        return NotImplemented

def _ensure_unary(obj, args):
    """
    Raises if args not a 1-tuple of obj
    """
    if len(args) != 1 or args[0] is not obj:
        raise NotImplementedError

# Note: by convention ufuncs use `*args` and functions use `args`
@ImplicitMatrix.implements(np.diagonal)
def _diagonal(obj, args):
    """
    Implements np.diagonal as a function call
    """
    _ensure_unary(obj, args)
    return obj.diagonal()

@ImplicitMatrix.implements(np.linalg.inv)
def _inv(obj, args):
    """
    Implements np.linalg.inv as a `inv` method call
    """
    _ensure_unary(obj, args)
    return obj.inv()

@ImplicitMatrix.implements(np.matmul)
def _matmul(obj, *args):
    """
    Limited implememtation of numpy.matmul
    """
    if len(args) != 2:
        return NotImplemented

    left, right = args

    if left is obj:
        return left.matmul_right(right)

    if right is obj:
        return right.matmul_left(left)

    return NotImplemented

@ImplicitMatrix.implements(np.add)
def _add(obj, *args):
    if len(args) != 2:
        return NotImplemented
    left, right = args
    if left is obj:
        return obj.add(right)
    if right is obj:
        return obj.add(left)
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

    def inv(self):
        """
        Inverse scalar times identity, raises if 0
        """
        return ScaledIdentity(scalar=1.0 / self.scalar)

    def add(self, other):
        """
        Concrete addition
        """
        # For now we only consider adding to a concrete array, in which case we
        # can concretize self
        dim = np.shape(other)[0]
        return other + self.scalar * np.eye(dim)

class Diagonal(ImplicitMatrix):
    """
    Implicit diagonal matrix
    """
    def __init__(self, diagonal):
        self._diagonal = diagonal

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
        return self._diagonal[:, None] * right


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

    def inv(self):
        """"
        Representation of inverse through Woodbury identity
        """
        inv_base = np.linalg.inv(self.base)
        inv_weight = np.linalg.inv(self.weight)
        capacitance = inv_weight + self.factor.T @ (inv_base @ self.factor)

        return RWSS(
            base=inv_base,
            factor=inv_base @ self.factor,
            weight=-np.linalg.inv(capacitance)
            )

def as_matrix(obj):
    """
    Performs conversion of scalars into implicit multiples of the identity matrix, and
    later 1-d arrays into implicit diagonal arrays.
    """
    # Implicit objects may not have a concrete shape, but they already matrices
    if isinstance(obj, ImplicitMatrix):
        return obj

    # This should work for a wide range of non-numpy objects
    shape = np.shape(obj)

    if len(shape) == 0:
        return ScaledIdentity(obj)
    if len(shape) == 1:
        return Diagonal(obj)
    if len(shape) == 2:
        return obj

    raise NotImplementedError
