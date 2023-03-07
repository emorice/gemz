"""
Abstract array glue

We need to deal with collections of array-like objects of different types,
especially our meta-arrays and jax arrays. They have close but distinct apis
so some intermediate layer is necessary, provided by the classes below
"""

import numpy as np

class ArrayAPI:
    """
    Abstract interface to a library implementing array-like objects
    """
    @classmethod
    def array_function(cls, func, *args, **kwargs):
        """
        Maps a numpy function to an equivalent and apply it
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def asarray(cls, obj):
        """
        Convert obj to an api array if necessary
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def broadcast_to(cls, obj, shape):
        """
        Create new object from existing by broadcasting
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def expand_dims(cls, array, axis):
        """
        Applies np.expand_dims
        """
        return cls.array_function(np.expand_dims, array, axis)

    @classmethod
    def zeros(cls, shape):
        """
        Create an array of specified shape full of zeros
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def eye(cls, length):
        """
        Create an identity squaure array of specified dim
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def inv(cls, array):
        """
        Applies np.linalg.inv
        """
        return cls.array_function(np.linalg.inv, array)

    @classmethod
    def slogdet(cls, array):
        """
        Applies np.linalg.inv
        """
        return cls.array_function(np.linalg.slogdet, array)

    @classmethod
    def solve(cls, array, rhs):
        """
        Applies np.linalg.solve
        """
        return cls.array_function(np.linalg.solve, array, rhs)
