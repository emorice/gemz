"""
Middleware to transparently use numpy functions on jax objects

Some code may be blind to the numpy backend used and use vanilla numpy functions
on its arguments. These are not supported by jax tracer objects. Hence, we wrap
jax tracers into numpy-aware objects that translate numpy function calls into jax
calls.
"""

import jax
import jax.numpy as jnp

# We whitelist operators one by one as we need them
_OP_NAMES = {
    # Unary
    '__neg__',
    # Binary, direct
    '__add__',
    '__mul__',
    '__matmul__',
    '__truediv__',
    '__pow__',
    # Binary, reverse
    '__radd__',
    '__rsub__',
    '__rmul__',
    '__rmatmul__',
    '__rtruediv__',
    }

class JaxObject:
    """
    Vanilla-numpy-compatible wrapper around jax arrays and tracers
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped

    # pylint: disable=no-self-use # numpy interface definition
    def __array_ufunc__(self, ufunc, method, *args):
        if method != "__call__":
            raise NotImplementedError
        ufunc_name = ufunc.__name__
        jax_ret = getattr(jnp, ufunc_name)(*maybe_unwrap_many(*args))
        return maybe_wrap(jax_ret)

    def __array_function__(self, func, _, args, kwargs):
        np_module = func.__module__
        if np_module == 'numpy':
            jax_module = jnp
        elif np_module == 'numpy.linalg':
            jax_module = jnp.linalg
        else:
            raise NotImplementedError

        func_name = func.__name__
        jax_func = getattr(jax_module, func_name)
        jax_ret = jax_func(
            *maybe_unwrap_many(*args),
            **maybe_unwrap_many_kw(**kwargs),
            )
        return maybe_wrap(jax_ret)

    def __getitem__(self, *indices):
        return maybe_wrap(self.wrapped.__getitem__(*indices))

    @property
    # pylint: disable=invalid-name # numpy interface
    def T(self):
        """
        Transpose
        """
        return maybe_wrap(self.wrapped.T)

def op_wrapper(name):
    """
    Delegate call to an operator by name
    """
    def _wrap(self, *args, **kwargs):
        jax_bound_func = getattr(self.wrapped, name)
        jax_ret = jax_bound_func(
            *maybe_unwrap_many(*args),
            **maybe_unwrap_many_kw(**kwargs),
            )
        return maybe_wrap(jax_ret)
    return _wrap

for _op_name in _OP_NAMES:
    setattr(JaxObject, _op_name, op_wrapper(_op_name))

def is_jax(obj):
    """
    Whether obj is a jax object
    """
    if isinstance(obj, jax.core.Tracer):
        return True
    return False

def maybe_wrap(obj):
    """
    Wrap object in JaxObject if necessary
    """
    if is_jax(obj):
        return JaxObject(obj)
    return obj

def maybe_unwrap(obj):
    """
    Extract from a JaxObject when needed
    """
    return (
        obj.wrapped
        if isinstance(obj, JaxObject)
        else obj
        )
def maybe_unwrap_many(*objs):
    """
    Extract from JaxObjects when needed
    """
    return map(maybe_unwrap, objs)

def maybe_unwrap_many_kw(**objs):
    """
    Extract from JaxObjects when needed
    """
    return {
        k: maybe_unwrap(obj)
        for k, obj in objs.items()
        }

def indirect_jax(function):
    """
    Function decorator to wrap jax arguments in numpy-compatible objects, and
    unwrap the return value if needed.

    This essentially makes pure numpy function jax-traceable.
    """
    def _wrap(*args, **kwargs):
        args = [ maybe_wrap(a) for a in args ]
        kwargs = { k: maybe_wrap(a) for k, a in kwargs.items() }
        wrapped_ret = function(*args, **kwargs)
        return maybe_unwrap(wrapped_ret)

    return _wrap
