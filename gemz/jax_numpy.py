"""
Middleware to transparently use numpy functions on jax objects

Some code may be blind to the numpy backend used and use vanilla numpy functions
on its arguments. These are not supported by jax tracer objects. Hence, we wrap
jax tracers into numpy-aware objects that translate numpy function calls into jax
calls.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special as jsc

from jax import custom_jvp

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

        jax_ufunc = getattr(jnp, ufunc_name, None)
        if jax_ufunc is None: # Try in scipy.special
            jax_ufunc = getattr(jsc, ufunc_name, None)
        if jax_ufunc is None:
            raise NotImplementedError(ufunc_name)

        return unjaxify(jax_ufunc)(*args)

    def __array_function__(self, func, _, args, kwargs):
        np_module = func.__module__

        # Special implementations
        if func is np.vstack:
            return vstack(*args, **kwargs)

        # Automatic wrapping
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

    def sum(self, *args, **kwargs):
        """
        Sum method
        """
        return maybe_wrap(self.wrapped.sum(*args, **kwargs))

    def __repr__(self):
        return (
            'JaxObject('
            + repr(self.wrapped)
            + ')'
            )

    @staticmethod
    def eager_map(function, *arrays):
        """
        Custom map-like function used to plug lax.map. Default implementation
        falls back to a loop.
        """
        @unjaxify(has_aux=True)
        def _imap(*_arrays):
            return _eager_map(
                    lambda args: jaxify(function, has_aux=True)(*args),
                    _arrays)

        return _imap(*arrays)

def vstack(arrays, **kwargs):
    """
    Unjaxified version of jnp.vstack.

    Cannot be handled by @unjaxify because the arrays are nested inside the
    first argument
    """
    return maybe_wrap(jnp.vstack([maybe_unwrap(a) for a in arrays], **kwargs))

@partial(custom_jvp, nondiff_argnums=(0,))
def _eager_map(function, arrays):
    """
    Wrapper of lax.map with custom derivative forcing eager backpropagation and
    computation of the jacobian of the op during the forward pass. Function must
    return a tuple of scalars.

    This decreases memory use when the mapped function has few dimensions of
    outputs, but shouldn't be used if the output is high-dimensional.
    """
    return jax.lax.map(function, arrays)

@_eager_map.defjvp
def _eager_map_jvp(function, primals, tangents):
    # notation:: tin: input tuple, map: dimension being mapped, tout: out tuple,
    #   ex: extra input dimensions, vout: vector with length matching tout
    arrays_tin_map_ex, = primals
    arrays_dot_tin_map_ex, = tangents

    def _item_value_and_jac(primals_tin_ex):
        # Eager backprop
        # ==============
        primals_tout, vjp_fun = jax.vjp(function, primals_tin_ex)
        # this assumes primals_tout is a tuple of scalars
        jac_tin_vout_ex, = jax.vmap(vjp_fun)(tuple(jnp.eye(len(primals_tout))))

        # At this point, vjp_fun and its bound residuals can be freed
        # jac_tin_vout_ex is a input-like tree of arrays with one dim per output
        jac_tout_tin_ex = tuple(zip(*(
                tuple(jac_vout_ex)
                for jac_vout_ex in jac_tin_vout_ex)))

        return primals_tout, jac_tout_tin_ex

    primals_tout_map, jac_tout_tin_map_ex = jax.lax.map(_item_value_and_jac,
            arrays_tin_map_ex)

    cotangents_tout = tuple( # tuple over output dims
        sum( # sum over input dims
            jax.vmap(jnp.sum)(array_dot_map_ex * jac_map_ex)
            for array_dot_map_ex, jac_map_ex
            in zip(arrays_dot_tin_map_ex, jac_tin_map_ex)
            )
        for jac_tin_map_ex in jac_tout_tin_map_ex
        )

    return primals_tout_map, cotangents_tout

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
