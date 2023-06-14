"""
Models wrapping an other model
"""

import numpy as np

from gemz import jax_utils
from gemz.jax_numpy import jaxify

from gemz.model import (
        Model, VstackTensorContainer, as_tensor_container,
        IndexTuple, as_index, Distribution, EachIndex
        )

class AddedConstantModel(Model):
    """
    Wraps an other model, adding a constant row to the data and conditionning
    automatically on it
    """
    def __init__(self, inner: Model):
        self.inner = inner

    def _condition(self, unobserved_indexes, data):
        _n_rows, n_cols = data.shape
        augmented_data = VstackTensorContainer((
            as_tensor_container(np.ones((1, n_cols))), data
            ))
        rows, cols = unobserved_indexes
        augmented_rows = IndexTuple((as_index(slice(0, 0)), rows))
        return self.inner._condition(
                (augmented_rows, cols),
                augmented_data)

class ScaledModel(Model):
    """
    Wraps an other model, scaling the data

    The logic is that if the wrapped model is some "standard" model, the scaled
    model should be appropriate for model with scale "scale". Thus, input data
    is divided by the scale to get the inner model data, and inner model
    predictions, conversely, get multiplied by the scale.
    """
    def __init__(self, inner: Model, scale = None):
        self.inner = inner
        self.scale = scale

    def _condition(self, unobserved_indexes, data, scale = None):
        if scale is None:
            scale = self.scale
        if scale is None:
            raise TypeError('Missing scale')

        cond = self.inner._condition(unobserved_indexes, data / scale)

        return Distribution(
                # Scaled
                mean=cond.mean * scale,
                # Unchanged under scaling
                sf_radial_observed=cond.sf_radial_observed,
                # Change of variable
                logpdf_observed=cond.logpdf_observed - cond.total_dims * np.log(scale),
                # Unchanged
                total_dims=cond.total_dims
                )

    def get_unbound_params(self):
        # TODO: propagate inner params
        if self.scale is None:
            return (
                    { 'scale': 1.0 },
                    { 'scale': jax_utils.RegExp() }
                    )
        return {}, {}

@jaxify
def logpdf(model, unobserved, data, **params):
    """
    Functional sum of conditional logpdfs
    """
    return model.conditional[unobserved](data, **params).logpdf_observed.sum()

class PlugInModel(Model):
    """
    Wraps an other model, optimizing out any unbound parameters
    """
    def __init__(self, inner: Model):
        self.inner = inner

    def _condition(self, unobserved_indexes, data):
        params_init, params_bijectors = self.inner.get_unbound_params()

        rows, _cols = unobserved_indexes

        # There is in theory a lot going on here, but for now only default
        # behavior for a small subset of configurations is actually implemented

        # Select training data
        #  * Ideally, either the union or intersection of all the observed data
        # across all conditioning patterns
        #  * For now, keep the observed rows and all columns
        training_data = data[~rows, :]

        # Marginalize model
        #  * Ideally, should ask the model to marginalize itself
        #  * For now, assume implicit marginalization: the marginal is assumed
        #  to be the same Model object when a subset of the data is passed to it
        marginal = self.inner

        # Training conditional pattern
        #  * Ideally, configurable
        #  * For now, alway use the pseudo-likelihood over columns
        training_cond = as_index(slice(None)), EachIndex

        max_results = jax_utils.maximize(
            logpdf,
            init=params_init,
            data={
                'model': marginal,
                'unobserved': training_cond,
                'data': training_data,
                },
            bijectors=params_bijectors,
            scipy_method='L-BFGS-B',
            )

        opt_params = max_results['opt']

        return self.inner._condition(unobserved_indexes, data, **opt_params)
