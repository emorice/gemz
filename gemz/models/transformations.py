"""
Models wrapping an other model
"""

from dataclasses import dataclass

import numpy as np

from gemz import jax_utils
from gemz.jax_numpy import jaxify

from gemz.model import (
        Model, TransformedModel, VstackTensorContainer, as_tensor_container,
        IndexTuple, as_index, Distribution, EachIndex
        )

class AddedConstantModel(TransformedModel):
    """
    Wraps an other model, adding a constant row to the data and conditionning
    automatically on it
    """
    def __init__(self, inner: Model, **params):
        super().__init__(inner)
        self.add_param('offset', jax_utils.RegExp(), 1.0)
        self.bind_params(**params)

    def _condition(self, unobserved_indexes, data, **params):
        this_params, inner_params = self._split_params(params)
        offset = self.get_params(**this_params)['offset']

        _n_rows, n_cols = data.shape
        augmented_data = VstackTensorContainer((
            as_tensor_container(offset * np.ones((1, n_cols))), data
            ))
        rows, cols = unobserved_indexes
        augmented_rows = IndexTuple((as_index(slice(0, 0)), rows))
        return self.inner._condition(
                (augmented_rows, cols),
                augmented_data, **inner_params)

class ScaledModel(TransformedModel):
    """
    Wraps an other model, scaling the data

    The logic is that if the wrapped model is some "standard" model, the scaled
    model should be appropriate for model with scale "scale". Thus, input data
    is divided by the scale to get the inner model data, and inner model
    predictions, conversely, get multiplied by the scale.
    """
    def __init__(self, inner: Model, mode='global', **params):
        super().__init__(inner)
        if mode == 'global':
            self.add_param('scale', jax_utils.RegExp(), 1.0)
        elif mode == 'column':
            self.add_param('scale', jax_utils.RegExp(),
                    lambda data: np.ones(np.shape(data[-1]))
                    )
        else:
            raise ValueError(f'Unknown scaling mode "{mode}"')
        self.mode = mode
        self.bind_params(**params)

    def _condition(self, unobserved_indexes, data, **params):
        this_params, inner_params = self._split_params(params)
        scale = self.get_params(**this_params)['scale']

        cond = self.inner._condition(unobserved_indexes, data / scale,
                **inner_params)

        return ScaledDistribution(cond, scale, mode=self.mode)

class GroupScaledModel(TransformedModel):
    """
    A ScaledModel where the scales are constrained to be identical inside
    pre-specified groups of variables
    """

class ScaledDistribution(Distribution):
    def __init__(self, inner, scale, mode):
        self.inner = inner
        self.scale = scale
        self.mode = mode

        # Unchanged
        self.total_dims = inner.total_dims

    @property
    def mean(self):
        # Scaled
        return self.inner.mean * self.scale

    @property
    def sf_radial_observed(self):
        # Unchanged under scaling
        return self.inner.sf_radial_observed

    @property
    def logpdf_observed(self):
        # Change of variable
        return self.inner.logpdf_observed - self.inner.total_dims * np.log(self.scale)
        # FIXME: this is valid only for the global case and for the column case
        # when doing LOO on columns. Note that in the loo case total_dims will
        # be the number of new rows.

@jaxify
def logpdf(model, unobserved, data, **params):
    """
    Functional sum of conditional logpdfs
    """
    return model.conditional[unobserved](data, **params).logpdf_observed.sum()

def get_training_data(data, unobserved_indexes):
    """
    Select training data

    For now, keep the observed rows and all columns.

    Fundamentally, this should simply be all the data that is marked as
    observed, but there are complications.
     * Many models can only deal with a perfect matrix of observations, without
        holes.
     * If we are sharing computations between leave-one-out models, excluding
        all observed data leaves us with nothing.
     * If we want to learn per-column or per-row parameters, all rows or columns
     needed have to be represented in the training data

    This calls for a complex and subtle way of choosing what should be used as
    "training data" in general, but for now we'll organically add sensible
    case-by-case definition.
    """
    rows, _cols = unobserved_indexes
    return data[~rows, :]

class PlugInModel(TransformedModel):
    """
    Wraps an other model, optimizing out any unbound parameters
    """
    def _condition(self, unobserved_indexes, data):
        params_init, params_bijectors = self.inner.get_unbound_params()

        # There is in theory a lot going on here, but for now only default
        # behavior for a small subset of configurations is actually implemented

        # Select training data
        training_data = get_training_data(data, unobserved_indexes)

        # Marginalize model
        #  * Ideally, should ask the model to marginalize itself
        #  * For now, assume implicit marginalization: the marginal is assumed
        #  to be the same Model object when a subset of the data is passed to it
        marginal = self.inner

        # Training conditional pattern
        #  * Ideally, configurable
        #  * For now, alway use the pseudo-likelihood over columns
        training_cond = as_index(slice(None)), EachIndex


        # Initialize parameters that depends on data or data shape
        params_init = {
                name: init(training_data) if callable(init) else init
                for name, init in params_init.items()
                }
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

        return PlugInDistribution(
                inner=self.inner._condition(unobserved_indexes, data,
                    **opt_params),
                opt_results=max_results,
                opt_init=params_init,
                objective_name='Negative pseudo log likelihood'
                )


@dataclass
class PlugInDistribution:
    inner: Distribution
    opt_results: dict
    opt_init: dict
    objective_name: str

    def as_dict(self):
        return self.inner.as_dict()

    def export_diagnostics(self, backend):
        return [
                backend.optimization_trace(
                    self.opt_results['hist'], self.objective_name
                    ),
                *backend.optimized_parameters(
                    self.opt_init,
                    self.opt_results['opt'],
                    )
                ]

    @property
    def mean(self):
        return self.inner.mean
