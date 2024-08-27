"""
Collection of functions to integrate jax autodiff with scipy solvers.

Convention for bijectors: `forward` maps anon parameters (unconstrained reals)
to natural parameters.
"""

import numpy as np
import scipy
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp

def pack(kwargs):
    """
    Flattens a tree of tensors into a single array.
    """
    leaves, struct = jax.tree_util.tree_flatten(kwargs)
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
    return jax.tree_util.tree_unflatten(struct, leaves)

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

def gen_obj(shapes, struct, bijs, native_obj, obj_mult, has_aux=False, jit=True):
    """
    Generates an objective funtion of a flat parameter vector from an
    keyword-based objective function

    Args:
        native_obj: an objective function accepting keyword arguments for all
            variable and fixed parameters, already constrained
        has_aux: whether native_obj returns a (obj, aux) tuple instead of obj.
        jit: whether to compile obj just-in-time

    Return:
        An objective function accepting a flat, unconstrained vector of
            variable, plus keyword arguments for all fixed parameters
    """
    def _obj(anon_pos, **obj_args):
        # Unflatten and transform working variables
        native_pos = apply_bijs(
                unpack(anon_pos, shapes, struct),
                bijs)
        # Call native obj, also with fixed variables
        value_aux = native_obj(**native_pos, **obj_args)

        # Multiply the result if needed
        if has_aux:
            value, aux = value_aux
            return obj_mult * value, aux
        return obj_mult * value_aux

    # Differentiate
    _diff_obj = jax.value_and_grad(_obj)

    # Optionally, jit
    if jit:
        return jax.jit(_diff_obj)
    return _diff_obj

def to_np64(obj, has_aux=False):
    """
    Generates a differentiated objective accepting and returning double numpy arrays
    from a differentiated objective accepting and returning float jax arrays
    """
    def _obj(np64_pos, **obj_args):
        j32_pos = jnp.array(np64_pos, dtype='float32')
        j32_va, j32_g = obj(j32_pos, **obj_args)
        if has_aux:
            # Unpack value-aux and convert only value
            j32_v, j32_a = j32_va
            np64_va = np.array(j32_v, dtype='float64'), j32_a
        else:
            # value-aux is just value, convert it
            np64_va = np.array(j32_va, dtype='float64')
        return np64_va, np.array(j32_g, dtype='float64')
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

def minimize(native_obj, init, data, bijectors=None, scipy_method=None,
    obj_mult=1., jit=True, has_aux=False, prog_desc=None):
    """
    High-level minimization function

    There are two very different ways to handle fixed (i.e., not subject to
    optimization) arguments of the objective function.
      * Bind the fixed parameters to the objective, e.g. as
        `lambda **kw: obj(myarg=myvalue, **kw)` before calling `minimize`. Such
        arguments will be accessible to JAX during tracing ; this is the
        strategy that should be used for loop bounds, complex objects, and
        generally anything that JAX complains about.
      * Pass the fixed parameters as key-values entries in `data`. Such
        parameters will be bound "just-in-time" *after* any tracing and compilation
        has occured. This is better suited for large arrays that you do not want
        JAX to constant-fold, for instance.

    Args:
        native_obj: an objective funtion accepting parameters and fixed values
            as keyword arguments
        init: a dictionary of initial values for the parameters to optimize
        data: a dictionary of fixed values for the other parameters
    """
    hist = []

    # Ensure arrays
    init = { k: jnp.array(v) for k,v in init.items() }
    # other args could be any type ?
    #data = { k: jnp.array(v) for k,v in data.items() }

    if bijectors is None:
        bijectors = {}
    # NOTE: this could be done in [un]apply bijs ?
    for k in init:
        if k not in bijectors:
            bijectors[k] = Identity()

    init_anon, shapes, struct = pack(unapply_bijs(init, bijectors))

    anon_obj = gen_obj(
        shapes, struct, bijectors,
        native_obj,
        obj_mult,
        has_aux=has_aux, jit=jit)

    _obj_scipy = to_np64(anon_obj, has_aux=has_aux)

    scipy_method = scipy_method or 'BFGS'

    if prog_desc is None:
        progress = None
    else:
        progress = tqdm(desc=prog_desc)

    def obj_scipy(flat_params):
        """
        Simple wrapper to record the objective values and bind fixed parameters
        """
        value, grad = _obj_scipy(flat_params, **data)
        # Also append aux to hist if present
        hist.append(value)
        # Then discard aux before yielding to scipy
        if has_aux:
            value, _ = value
        if progress is not None:
            progress.update()
        return value, grad

    if len(init_anon) > 0:
        opt = scipy.optimize.minimize(
            obj_scipy,
            init_anon,
            method=scipy_method,
            jac=True,
        )
    else:
        # It could be that after reparametrizing parameters to handle
        # constraints, you're left with zero degrees of freedom to optimize.
        # The scipy optimizer doesn't like that, so handle it separately.
        opt = {
            'x': init_anon
            }

    if progress is not None:
        progress.close()

    nat_opt = apply_bijs(unpack(opt['x'], shapes, struct), bijectors)

    return {
        'opt': nat_opt,
        'hist': hist,
        'scipy_opt': opt
        }

def maximize(native_obj, init, data, bijectors=None, scipy_method=None,
    obj_mult=1., jit=True, has_aux=False, prog_desc=None):
    """
    Counterpart to minimize
    """
    return minimize(
        native_obj, init, data,
        bijectors=bijectors, scipy_method=scipy_method,
        obj_mult=-obj_mult,
        jit=jit,
        has_aux=has_aux,
        prog_desc=prog_desc,
        )

class Bijector:
    """
    Abstract bijector class
    """
    def forward(self, free):
        """
        Maps free parameters to natural parameters
        """
        raise NotImplementedError

    def inverse(self, natural):
        """
        Inverse of forward
        """
        raise NotImplementedError

class Softmax(Bijector):
    """
    Softmax bijector mapping D logits to D+1 normalized weights in (0, 1).

    Allows batching over the *last* dimensions.
    """
    def forward(self, logits):
        """
        Maps logits -> weights
        """
        full_logits = jnp.insert(logits, 0, 0., axis=0)
        return jnp.exp(jax.nn.log_softmax(full_logits, axis=0))

    def inverse(self, weights):
        """
        Maps weights -> logits
        """
        weight0 = weights[0]
        weights = weights[1:]
        odd_ratios = weights / weight0
        return jnp.log(odd_ratios)

class Exp(Bijector):
    """
    Exp bijector.
    """
    def forward(self, free):
        """
        Maps real -> exp
        """
        return jnp.exp(free)

    def inverse(self, natural):
        """
        Maps postive reals -> log
        """
        return jnp.log(natural)

class RegExp(Bijector):
    """
    Regularized exp bijector.
    """
    def __init__(self, lower=0.):
        self.lower = lower

    def forward(self, inputs):
        """
        Maps real -> exp
        """
        return self.lower + jnp.exp(inputs)

    def inverse(self, outputs):
        """
        Maps postive reals -> log
        """
        return jnp.log(outputs - self.lower)

class Identity(Bijector):
    """
    Identity bijector
    """
    def forward(self, value):
        """
        x -> x
        """
        return value

    def inverse(self, value):
        """
        x -> x
        """
        return value
