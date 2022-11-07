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
# (because I'm not sure where to find a comprehensive list)
_OP_NAMES = {
    # Unary
    '__neg__',
    # Binary, direct
    '__add__',
    '__sub__',
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

    def __array_ufunc__(self, ufunc, method, *args):
        if method != "__call__":
            raise NotImplementedError
        ufunc_name = ufunc.__name__
        return unjaxify(getattr(jnp, ufunc_name))(*args)

    def __array_function__(self, func, _, args, kwargs):
        np_module = func.__module__
        if np_module == 'numpy':
            jax_module = jnp
        elif np_module == 'numpy.linalg':
            jax_module = jnp.linalg
        else:
            raise NotImplementedError

        func_name = func.__name__
        jax_func = unjaxify(getattr(jax_module, func_name))
        return jax_func(*args, **kwargs)

    def __getitem__(self, *indices):
        return maybe_wrap(self.wrapped.__getitem__(*indices))

    @property
    # pylint: disable=invalid-name # numpy interface
    def T(self):
        """
        Transpose
        """
        return maybe_wrap(self.wrapped.T)

    @property
    def shape(self):
        """
        Shape of the wrapped object.

        The returned shape is not itself wrapped, this is meant to be used for
        debugging, not reused as an input for further computations.
        """
        return self.wrapped.shape

    def __repr__(self):
        return (
            'JaxObject('
            + repr(self.wrapped)
            + ')'
            )

    @staticmethod
    def imap(function, *arrays):
        """
        Custom map-like function used to plug lax.map. Default implementation
        falls back to a loop.
        """
        @unjaxify(has_aux=True)
        def _imap(*_arrays):
            return jax.lax.map(
                    lambda args: jaxify(function, has_aux=True)(*args),
                    _arrays)

        return _imap(*arrays)

def op_wrapper(name):
    """
    Delegate call to an operator by name
    """
    def _wrap(self, *args, **kwargs):
        jax_bound_func = unjaxify(getattr(self.wrapped, name))
        return jax_bound_func(*args, **kwargs)
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

def jaxify(*function, has_aux=False):
    """
    Function decorator to wrap jax arguments in numpy-compatible objects, and
    unwrap the return value if needed.

    This essentially makes pure numpy function jax-traceable.

    Args:
        has_aux: for now limited to a single aux jax object
    """
    if not function:
        return lambda function: jaxify(function, has_aux=has_aux)

    assert len(function) == 1
    function = function[0]

    def _wrap(*args, **kwargs):
        args = [ maybe_wrap(a) for a in args ]
        kwargs = { k: maybe_wrap(a) for k, a in kwargs.items() }
        wrapped_ret = function(*args, **kwargs)
        if has_aux:
            return tuple(maybe_unwrap(item) for item in wrapped_ret)
        return maybe_unwrap(wrapped_ret)

    return _wrap

def unjaxify(*function, has_aux=False):
    """
    Inverse of jaxify: unwraps numpy-compatible jax wrappers before calling
    function and re-wraps the result.

    Used to write jax implementations of ops: it transforms a function operating
    on jax arrays into a function that can act on numpy-compatible JaxObjects.

    Args:
        has_aux: True if `function` returns a tuple of jax arrays that need to
            be wrapped individually. Else (default), the result is treated as
            one jax array and wrapped as a whole.
    """
    if not function:
        return lambda function: unjaxify(function, has_aux=has_aux)

    assert len(function) == 1
    function = function[0]

    def _wrap(*args, **kwargs):
        args = [ maybe_unwrap(a) for a in args ]
        kwargs = { k: maybe_unwrap(a) for k, a in kwargs.items() }
        unwrapped_ret = function(*args, **kwargs)
        if has_aux:
            return tuple(maybe_wrap(item) for item in unwrapped_ret)
        return maybe_wrap(unwrapped_ret)

    return _wrap
