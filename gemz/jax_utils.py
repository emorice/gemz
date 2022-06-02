"""
Collection of functions to integrate jax autodiff with scipy solvers.
"""

import numpy as np
import scipy

import distrax
import jax
import jax.numpy as jnp

def pack(kwargs):
    """
    Flattens a tree of tensors into a single array.
    """
    leaves, struct = jax.tree_flatten(kwargs)
    shapes = [leave.shape for leave in leaves]
    return (
        jnp.hstack([leave.flatten() for leave in leaves]),
        shapes,
        struct
    )

def unpack(data, shapes, struct):
    """
    Inverse of pack
    """
    i = 0
    leaves = []
    for shape in shapes:
        size = 1
        for dim in shape:
            size *= dim
        leaves.append(data[i:i+size].reshape(shape))
        i += size
    return jax.tree_unflatten(struct, leaves)

def apply_bijs(anon_tree, bijs):
    """
    Apply a tree of bijectors to a tree of tensors
    """
    return jax.tree_map(lambda leave, bij: bij.forward(leave), anon_tree, bijs)

def unapply_bijs(nat_tree, bijs):
    """
    Apply in reverse a tree of bijectors to a tree of tensors
    """
    return jax.tree_map(lambda leave, bij: bij.inverse(leave), nat_tree, bijs)

def gen_obj(shapes, struct, bijs, native_obj):
    """
    Generates an objective funtion of a flat parameter vector from an
    keyword-based objective function
    """
    return lambda data, **obj_args : jax.value_and_grad(
        lambda data: native_obj(
            **apply_bijs(
                unpack(data, shapes, struct),
                bijs),
            **obj_args
        )
    )(data)

def to_np64(obj):
    """
    Generates a differentiated objective accepting and returning double arrays
    from a differentiated objective accepting and returning float arrays
    """
    def _obj(np64_pos):
        j32_pos = jnp.array(np64_pos, dtype='float32')
        j32_v, j32_g = obj(j32_pos)
        return np.array(j32_v, dtype='float64'), np.array(j32_g, dtype='float64')
    return _obj

def vj_argnames(function, names):
    """
    Kind of value_and_jac that also accepts selection of AD params by name and not position
    """
    def split_function(ad_params, nonad_params):
        ad_params.update(**nonad_params)
        return function(**ad_params)
    def _vg(**params):
        # Split
        ad_params = {k: params[k] for k in names}
        for k in names:
            del params[k]
        value, pullback, aux = jax.vjp(
            lambda ad_params: split_function(ad_params, params),
            ad_params,
            has_aux=True
        )
        # Non-trivial code with no public API yet
        # See https://github.com/google/jax/discussions/10081
        # pylint: disable=protected-access
        jac = jax.vmap(pullback)(jax._src.api._std_basis(value))
        jac = jax.tree_transpose(
            jax.tree_structure(ad_params),
            jax.tree_structure(value),
            jax.tree_map(
                lambda leaf: jax.tree_unflatten(jax.tree_structure(value), leaf), jac[0])
                  )
        return value, jac, aux
    return _vg

def minimize(native_obj, init, data, scipy_method=None, obj_mult=1., jit=None):
    """
    High-level minimization function

    Args:
        native_obj: an objective funtion accepting parameters and fixed values
            as keyword arguments
        init: a dictionary of initial values for the parameters to optimize
        data: a dictionary of fixed values for the other parameters
    """
    hist = []

    # Ensure arrays
    init = { k: jnp.array(v) for k,v in init.items() }
    data = { k: jnp.array(v) for k,v in data.items() }

    # For now we put defaults for compatibilty, later we will expose this
    bijectors = { k: distrax.Lambda(lambda x: x) for k in init }

    init_anon, shapes, struct = pack(unapply_bijs(init, bijectors))

    anon_obj = gen_obj(shapes, struct, bijectors, lambda **kw: obj_mult * native_obj(**kw, **data))

    _obj_scipy = to_np64(jax.jit(anon_obj) if jit else anon_obj)

    scipy_method = scipy_method or 'BFGS'

    def obj_scipy(flat_params):
        """
        Simple wrapper to record the objective values
        """
        value, grad = _obj_scipy(flat_params)
        hist.append(value)
        return value, grad

    opt = scipy.optimize.minimize(
        obj_scipy,
        init_anon,
        method=scipy_method,
        jac=True,
    )

    nat_opt = apply_bijs(unpack(opt['x'], shapes, struct), bijectors)

    return {
        'opt': nat_opt,
        'hist': hist,
        'scipy_opt': opt
        }

def maximize(native_obj, init, data, scipy_method=None, obj_mult=1., jit=None):
    """
    Counterpart to minimize
    """
    return minimize(
        native_obj, init, data, scipy_method=scipy_method,
        obj_mult=-obj_mult,
        jit=jit
        )
