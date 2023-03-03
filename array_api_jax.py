"""
Jax provider for array api
"""

import numpy as np
import jax.numpy as jnp

from array_api import ArrayAPI


class JaxAPI(ArrayAPI):
    """
    Conversion layer from numpy to jax operations
    """
    functions = {
        np.diagonal: jnp.diagonal,
        np.outer: jnp.outer,
        np.linalg.inv: jnp.linalg.inv,
        np.linalg.slogdet: jnp.linalg.slogdet,
        np.linalg.solve: jnp.linalg.solve,
        }

    @classmethod
    def array_function(cls, func, *args, **kwargs):
        """
        Maps a numpy function to an equivalent and apply it
        """
        if func in cls.functions:
            return cls.functions[func](*args, **kwargs)
        raise NotImplementedError(func)

    @classmethod
    def asarray(cls, obj):
        return jnp.asarray(obj)

    @classmethod
    def broadcast_to(cls, obj, shape):
        return jnp.broadcast_to(obj, shape)

    @classmethod
    def zeros(cls, shape):
        return jnp.zeros(shape)

    @classmethod
    def eye(cls, length):
        return jnp.eye(length)

class MetaJaxAPI(JaxAPI):
    """
    Dymamic conversion to jax that tries to use numpy protocol first

    Array creation defaults to jax types.
    """
    @classmethod
    def array_function(cls, func, *args, **kwargs):
        """
        Apply function as-is if first positional argument defines
        an __array_function__, else try to use jax equivalent.
        """
        if args and hasattr(args[0], '__array_function__'):
            return func(*args, **kwargs)
        return super().array_function(func, *args, **kwargs)
