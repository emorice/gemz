"""
Abstract model interface
"""

import importlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gemz.models import methods
from gemz.models.methods import ModelSpec

# Indexing utils
class Index:
    """
    Abstract index-like class
    """
    def to_mask(self, length):
        """
        Convert self to binary mask
        """
        raise NotImplementedError

    def __invert__(self):
        return InvIndex(self)

@dataclass
class SliceIndex(Index):
    """
    Slice-based indexing
    """
    py_slice: slice
    def to_mask(self, length):
        indexes = np.arange(length)

        if self.py_slice.step is not None:
            raise NotImplementedError
        start = self.py_slice.start
        if start is None:
            start = 0
        stop = self.py_slice.stop
        if stop is None:
            stop = length

        return (indexes >= start) & (indexes < stop)

class _EachIndexT(Index):
    """
    Special indexing constant type meant to be used as singleton

    When used to condition a distribution, indicates that the distribution should be
    conditioned on all but each index of the considered axis in turn, yielding a
    family of conditional distributions
    """
    def __repr__(self):
        return 'EachIndex'

EachIndex = _EachIndexT()

IndexLike = Index | slice | int

def as_index(index_like: IndexLike) -> Index:
    """
    Wrap index_like object into an index
    """
    if isinstance(index_like, Index):
        return index_like

    if isinstance(index_like, slice):
        return SliceIndex(index_like)
    if isinstance(index_like, int):
        if index_like >= 0:
            return SliceIndex(slice(index_like, index_like+1))

    raise NotImplementedError(index_like)

@dataclass
class InvIndex(Index):
    """
    Logical inverse (complement) of an other index
    """
    index: Index

    def to_mask(self, length):
        return ~self.index.to_mask(length)

    def __invert__(self):
        return self.index

class IndexTuple(Index):
    """
    Tuple of other indexes, meant to index logically stacked tensors
    """
    def __init__(self, index_likes: Iterable[IndexLike]):
        self.indexes = tuple(map(as_index, index_likes))

    def __invert__(self) -> 'IndexTuple':
        return IndexTuple(tuple(
            ~index for index in self.indexes
            ))

class TensorContainer:
    """
    Container object meant to work with indexes
    """
    def __getitem__(self, indexes: tuple[Index, ...]):
        raise NotImplementedError

TensorContainerLike =  TensorContainer | NDArray

class SingleTensorContainer(TensorContainer):
    """
    Container for a single Tensor-like object, extends fancy indexing to work
    with Index objects
    """
    def __init__(self, tensor: NDArray):
        self.tensor = tensor

    def __getitem__(self, indexes) -> NDArray:
        masks = tuple(
                as_index(ind_like).to_mask(length)
                for ind_like, length in zip(indexes, np.shape(self.tensor))
                )
        return self.tensor[np.ix_(*masks)]

    @property
    def shape(self):
        return self.tensor.shape

    def __mul__(self, other):
        return SingleTensorContainer(self.tensor * other)

    def __truediv__(self, other):
        return SingleTensorContainer(self.tensor / other)

class VstackTensorContainer(TensorContainer):
    """
    Container representing several other containers, vertically stacked
    """
    def __init__(self, containers: Iterable[ArrayLike]):
        self.containers = tuple(map(as_tensor_container, containers))
        n_cols = {cont.shape[-1] for cont in self.containers}
        if not n_cols:
            self._n_cols = None
        elif len(n_cols) == 1:
            self._n_cols = next(iter(n_cols))
        else:
            raise ValueError('Vertically stacked tensor containers must have '
                f'matching column numbers, got {n_cols}.')

    def __getitem__(self, indexes) -> NDArray:
        if len(indexes) != 2:
            raise NotImplementedError
        rows, cols = indexes
        if not isinstance(rows, IndexTuple):
            raise ValueError('Row index for vertical tensor stack must be '
                'IndexTuple, got ' + repr(rows))
        arrays = [ container[cont_rows, cols]
                for container, cont_rows in zip(self.containers, rows.indexes) ]
        return np.vstack(arrays)

    @property
    def shape(self):
        return None, self._n_cols

def as_tensor_container(tensor_like: TensorContainerLike):
    """
    Wrap object into a tensor container
    """
    if isinstance(tensor_like, TensorContainer):
        return tensor_like
    return  SingleTensorContainer(tensor_like)

# Model

MODULES = {
    'linear': 'gemz.models.linear',
    'mt_std': 'gemz.models.mt_sym',
    'mt_sym': 'gemz.models.mt_sym',
    'mt_het': 'gemz.models.mt_sym',
    'clmt': 'gemz.models.clmt',
    }
"""
Dictionary of modules defining the corresponding named model for lazy loading
"""

class Model:
    """
    Unified model interface
    """
    @classmethod
    def from_spec(cls, spec: ModelSpec) -> 'Model':
        """
        Instantiate model from a specification document

        The module defining the model is lazily loaded here.
        """
        if 'model' not in spec:
            raise ValueError('Invalid spec: spec must contain a "model" key')
        module_name = MODULES.get(spec['model'])
        if not module_name:
            return FitPredictCompat.from_spec(spec)

        module = importlib.import_module(module_name)

        return module.make_model(spec)

    @property
    def conditional(self):
        """
        Return an indexable and callable object for convenient specification of conditional
        tasks
        """
        return ConditionMaker(self)

    def _condition(self, unobserved_indexes, data, **params):
        """
        Model-specific implementation of conditionals.

        This is the most central method to implement in subclasses
        """
        raise NotImplementedError

    def __init__(self):
        self.parameters = []
        self.bijectors = {}
        self.init = {}
        self.values = {}

    def add_param(self, name, bijector=None, init=None):
        """
        Register a model parameter
        """
        self.parameters.append(name)
        self.bijectors[name] = bijector
        self.init[name] = init

    def bind_params(self, **params):
        """
        Store value of parameters that have been registered first
        """
        self.ensure_declared(params)
        self.values |= params

    def ensure_declared(self, param_names):
        """
        Raise if param_names contains an undeclared parameter
        """
        for name in param_names:
            if name not in self.parameters:
                raise TypeError(f'No such parameter: \'{name}\''
                        + f' (valid parameters: {self.parameters})')

    def get_local_unbound_params(self):
        """
        Get bijectors and initial values for all declared parameters that haven't
        been bound yet. Only returns this object's parameters.
        """
        bijs, inits = {}, {}
        for name in self.parameters:
            if name not in self.values:
                bijs[name] = self.bijectors[name]
                inits[name] = self.init[name]
        return inits, bijs

    def get_unbound_params(self):
        """
        Get bijectors and initial values for all declared parameters that haven't
        been bound yet. Returns this object and all nested objects' parameters.
        """
        return self.get_local_unbound_params()

    def get_params(self, **params):
        """
        Merge given params with bound values, raising on both missing and extra
        params
        """
        self.ensure_declared(params)
        merged = self.values | params
        for name in self.parameters:
            if name not in merged:
                raise TypeError(f'Missing parameter: {name}')
        return merged

class FinalModel(Model):
    """
    Model providing concrete numerical implementations of conditionals on a case
    disjunction basis
    """
    def _condition(self, unobserved_indexes, data, **params):
        """
        Model-specific implementation of conditionals

        This attempts to use pattern-specific implementations if available, and
        raises if they are not. If a subclass has a generic implementation, it
        should either try it before calling super or catch the
        NotImplementedError raised by super.
        """
        if len(unobserved_indexes) == 2:
            ind0, ind1 = unobserved_indexes
            if ind0 is EachIndex:
                if ind1 is EachIndex:
                    return self._condition_loo_loo(unobserved_indexes, data,
                            **params)
                return self._condition_loo_block(unobserved_indexes, data,
                        **params)
            if ind1 is EachIndex:
                return self._condition_block_loo(unobserved_indexes, data,
                        **params)
            return self._condition_block_block(unobserved_indexes, data,
                    **params)
        raise NotImplementedError

    def _condition_loo_loo(self, unobserved_indexes, data, **params):
        """
        Specialized conditionner for LOO conditioning on both axes of a matrix
        distribution
        """
        raise NotImplementedError

    def _condition_block_loo(self, unobserved_indexes, data, **params):
        """
        Specialized conditionner for LOO conditioning on the second axis of a
        matrix distribution
        """
        raise NotImplementedError

    def _condition_loo_block(self, unobserved_indexes, data, **params):
        """
        Specialized conditionner for LOO conditioning on the first axis of a
        matrix distribution
        """
        raise NotImplementedError

    def _condition_block_block(self, unobserved_indexes, data, **params):
        """
        Specialized conditionner for non-loo conditioning of a
        matrix distribution
        """
        raise NotImplementedError

class TransformedModel(Model):
    """
    Any model that wraps an other model.

    This mixin takes care of propagating parameters up and down the model stack
    """
    def __init__(self, inner: Model):
        self.inner = inner
        super().__init__()

    def get_unbound_params(self):
        # Fixme: decorate names to avoid clashes
        return tuple( cur | inner for cur, inner in
                zip(self.get_local_unbound_params(),
                    self.inner.get_unbound_params())
                )

    def _split_params(self, params):
        """
        Split a dictionnary of parameters in two, separating this object's
        parameters from the inner model's parameters
        """
        # Fixme: decorate names to avoid clashes
        this_params, inner_params = {}, {}
        for name, value in params.items():
            if name in self.parameters:
                this_params[name] = value
            else:
                inner_params[name] = value
        return this_params, inner_params

    def bind_params(self, **params):
        """
        Store parameter values in either this model or a wrapped model

        Bugs: this does not display the full list of parameters on error but
        only the parameters of the innermost model
        """
        this_params, inner_params = self._split_params(params)
        super().bind_params(**this_params)
        self.inner.bind_params(**inner_params)

@dataclass
class ConditionMaker:
    """
    Indexable object to easily condition a model
    """
    model: Model

    def __getitem__(self, unobserved_indexes):
        """
        Convenience syntax to condition on slices
        """
        def condition_bound(data, **params):
            return self(unobserved_indexes, data, **params)
        return condition_bound

    def __call__(self, unobserved_indexes, data, **params):
        """
        Calling the condition property acts as if condition was a regular
        method
        """
        # TODO: wrap
        return self.model._condition(
                tuple(as_index(index_like) for index_like in unobserved_indexes),
                as_tensor_container(data),
                **params
                )

@dataclass
class Distribution:
    """
    Probabilistic information about a variable

    Attributes:
        mean: expected value of the variable
        observed: actual observed value. This is of course not always known, but
            it is convenient to have a field to store the observed value for the
            cases where it is known.
        var: to be defined/renamed
        total_dims: product of the dimensions spanned by the distribution
    """
    mean: Any = None
    observed: Any = None
    sf_radial_observed: Any = None
    # We use nan such that applying transformation formulas to pdf needs not
    # differentiating None / non-None case
    logpdf_observed: Any = np.nan
    total_dims: Any = None

    def sf_radial(self, observed=None):
        """
        Survival function function.
        """
        if observed is None:
            return self.sf_radial_observed
        raise NotImplementedError

    def logpdf(self, observed=None):
        """
        Log density function
        """
        if observed is None:
            return self.logpdf_observed
        raise NotImplementedError

    def as_dict(self):
        """
        Dictionary of constant attributes
        """
        return {
            'mean': self.mean,
            'sf_radial': self.sf_radial_observed,
            'logpdf': self.logpdf_observed,
            }

    def export_diagnostics(self, backend):
        del backend
        return []

    def metric_observed(self, metric_name: str, observed: NDArray | None = None):
        """
        Compute a measure of how well the observed data fits the distribution
        """
        if observed is None:
            observed = self.observed

        # RSS: arithmetic aggregation of columns
        # GEOM: geometric aggregation of columns
        # iRSS: no aggregation, returns each column
        squares = (observed - self.mean)**2
        if metric_name == 'RSS':
            return np.sum(squares)

        col_squares = np.sum(squares, 0)
        if metric_name == 'GEOM':
            # Note: the axes are swapped compared to the historical
            # implementation in cv.py. I currently believe that this was a
            # mistake in the later that never was detected.
            return np.sum(np.log(col_squares))
        if metric_name == 'iRSS':
            return col_squares

        raise ValueError(f'No such metric: {metric_name}')

class FitPredictCompat(Model):
    """
    Compatibility layer with old fit-predict interface
    """
    def __init__(self, method, spec: ModelSpec):
        super().__init__()
        self.method = method
        self.spec = spec

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> 'Model':
        """
        Generate a Model wrapper for older methods that exposed "fit" and
        "predict_loo" functions.

        spec is assumed to contain a "model" key.
        """

        method = methods.get(spec["model"])
        if method is None:
            raise NotImplementedError(f'No such model: {spec["model"]}')

        return cls(method, spec)

    def _condition(self, unobserved_indexes, data, **params):
        rows, cols = unobserved_indexes
        if cols is not EachIndex:
            raise ValueError('Legacy models only support leave-one-column-out'
                    ' patterns')
        train = data[~rows, :]
        test = data[rows, :]

        # Copied from deprecated fit opt
        # ==============================

        kwargs = dict(self.spec)
        del kwargs['model']

        # FIXME: this is not implemented in Model interface yet
        #if hasattr(model, 'OPS_AWARE') and model.OPS_AWARE:
        #    kwargs['_ops'] = _ops

        fitted = self.method.fit(train, **kwargs)

        predictions = self.method.predict_loo(fitted, test)
        
        return Distribution(mean=predictions)
