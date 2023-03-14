"""
Base code for array-like duck objects
"""
from typing import Type

import numpy as np

from .array_api import ArrayAPI
from .numpy_array_api import NumpyArrayAPI

class VirtualArray:
    """
    Base class for integration of virtual arrays with regular numpy code.

    Implements numpy protocol and more numpy-like apis, and dispatch them to a
    set of public and private methods to be implemented by base classes.
    """
    _implementations : dict = {}

    # Tensor library backend, defaults to numpy but meant to be overriden in
    # subclasses
    aa: Type[ArrayAPI] = NumpyArrayAPI

    @classmethod
    def implements(cls, func):
        """
        Function decorator to register an implementation of a numpy function or
        ufunc
        """
        def _wrapper(func_imp):
            cls._implementations[func] = func_imp
            return func_imp
        return _wrapper

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if ufunc in self._implementations and method == '__call__':
            imp = self._implementations[ufunc]
            return imp(self, *args, **kwargs)
        return NotImplemented

    def __array_function__(self, func, _, args, kwargs):
        if func in self._implementations:
            imp = self._implementations[func]
            return imp(self, args)
        return NotImplemented

    def __array__(self):
        # First densify using the object's backend, then convert to numpy
        return np.array(self._as_dense())

    @classmethod
    def as_dense(cls, array):
        """
        Convert array to dense if necessary. Actual type of array depends on
        backend
        """
        if isinstance(array, VirtualArray):
            return array._as_dense()
        return cls.aa.asarray(array)

    def _as_dense(self):
        raise NotImplementedError

    @property
    def ndim(self):
        """
        Number of dimensions
        """
        return len(self.shape)

    @property
    def shape(self):
        """
        Array shape
        """
        raise NotImplementedError

    def __add__(self, right):
        """
        Self + right
        """
        raise NotImplementedError

    def __radd__(self, left):
        """
        Left + self

        Note: both __add__ and __radd__ map to the same op, as we assume addition on
        ImplicitMatrices commutes. The distinction is mostly useful for
        types where '+' means concatenation (non commutative) I believe.
        """
        return self.__add__(left)

    # Todo: this will most likely become a general pattern to factor out
    @classmethod
    def broadcast_to(cls, array, shape):
        """
        Broadcast possibly virtual array to shape
        """
        # If virtual, dispatch to private implementation
        if isinstance(array, VirtualArray):
            return array._broadcast_to(shape)
        # Else, dispatch to backend
        return cls.aa.broadcast_to(array, shape)

    def _broadcast_to(self, shape):
        """
        Broadcast implementation
        """
        raise NotImplementedError

def _ensure_unary(obj, args):
    """
    Raises if args not a 1-tuple of obj
    """
    if len(args) != 1 or args[0] is not obj:
        raise NotImplementedError

# Note: by convention ufuncs use `*args` and functions use `args`
@VirtualArray.implements(np.diagonal)
def _diagonal(obj, args):
    """
    Implements np.diagonal as a function call
    """
    _ensure_unary(obj, args)
    return obj.diagonal()

@VirtualArray.implements(np.linalg.inv)
def _inv(obj, args):
    """
    Implements np.linalg.inv as a `inv` method call
    """
    _ensure_unary(obj, args)
    return obj.inv()

@VirtualArray.implements(np.linalg.solve)
def _solve(obj, args):
    """
    Implements np.linalg.inv as a `_solve` method call
    """
    _obj, *args = args
    assert _obj is obj
    return obj._solve(*args)

@VirtualArray.implements(np.linalg.slogdet)
def _inv(obj, args):
    """
    Implements np.linalg.sloget as a `slogdet` method call
    """
    _ensure_unary(obj, args)
    return obj.slogdet()

@VirtualArray.implements(np.matmul)
def _matmul(obj, *args):
    """
    Limited implememtation of numpy.matmul
    """
    if len(args) != 2:
        return NotImplemented

    left, right = args

    # If matmul was called on right object, reverse priorities and try right's
    # rmatmul before left's matmul
    if right is obj:
        return obj.__rmatmul__(left)

    return left @ right

@VirtualArray.implements(np.add)
def _add(obj, *args, out=None):
    if len(args) != 2:
        return NotImplemented
    left, right = args
    if left is obj:
        return obj.__add__(right)
    if right is obj:
        return obj.__add__(left)
    return NotImplemented

@VirtualArray.implements(np.ndim)
def _ndim(obj, args):
    _ensure_unary(obj, args)
    return obj.ndim

@VirtualArray.implements(np.shape)
def _shape(obj, args):
    _ensure_unary(obj, args)
    return obj.shape

# Legacy alias
ImplicitMatrix = VirtualArray
