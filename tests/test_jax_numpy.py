"""
Test custom jax layer
"""

import jax
import jax.numpy as jnp

from gemz import jax_numpy

def test_eager_map():
    """
    Test eager backprop
    """

    def foo(args):
        x, y = args
        return x * jnp.cos(y), x * jnp.sin(y)

    def ref_loop(X, Y):
        X_out, Y_out = jax.lax.map(foo, (X, Y))
        return jnp.sum(X_out) + jnp.sum(Y_out)

    _X = jnp.array([-2., 0., 1.])
    _Y = jnp.array([-0.5, 1.5, 0.])

    ref_val, ref_grad = jax.value_and_grad(ref_loop)(_X, _Y)

    def eager_loop(X, Y):
        X_out, Y_out = jax_numpy._eager_map(foo, (X, Y))
        return jnp.sum(X_out) + jnp.sum(Y_out)

    eager_val, eager_grad = jax.value_and_grad(eager_loop)(_X, _Y)

    assert jnp.allclose(eager_val, ref_val)
    assert jnp.allclose(eager_grad, ref_grad)
