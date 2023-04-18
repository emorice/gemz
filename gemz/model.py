"""
Abstract model interface
"""
from dataclasses import dataclass
import importlib

import numpy as np

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

    @classmethod
    def from_any(cls, index_like):
        """
        Convert from any index-like object commonly used
        """
        if isinstance(index_like, slice):
            return cls(index_like)
        if isinstance(index_like, int):
            if index_like >= 0:
                return cls(slice(index_like, index_like+1))
        if isinstance(index_like, Index):
            return index_like
        raise NotImplementedError(index_like)

@dataclass
class NegIndex(Index):
    """
    Logical negation of an other index
    """
    index: Index

    def to_mask(self, length):
        return ~self.index.to_mask(length)

@dataclass
class Conditioner:
    """
    Specification of a conditional pattern
    """
    indexes: tuple[Index, ...]

    @classmethod
    def from_indexes(cls, indexes: tuple):
        """
        Normalize indexes to slices
        """
        return cls(tuple(SliceIndex.from_any(ind)
                for ind in indexes
                ))

    def complement(self, axis=None):
        """
        Generate new conditionner with slices complemented.
        Reverses all axes by default
        """
        return self.from_indexes(tuple(
            NegIndex(ind) if axis in (i, None)
            else ind
            for i, ind in enumerate(self.indexes)
            ))

    def select(self, data):
        """
        Subset data

        For now, this is implemented with binary masks. This allows the
        operation to be completely generic but also memory-inefficient for
        subsets that are simple slices.
        """
        masks = tuple(ind.to_mask(length) for ind, length in zip(self.indexes,
            np.shape(data)))
        ans = data[np.ix_(*masks)]
        print(ans)
        return ans

# Model

class Model:
    """
    Abstract model interface
    """
    def __init__(self, spec: ModelSpec, conditioner: Conditioner|None = None):
        self.spec = spec
        self.conditioner = conditioner

    @classmethod
    def from_spec(cls, spec: ModelSpec):
        """
        Instantiate model from a specification document

        The module defining the model is lazily loaded here.
        """
        if 'model' not in spec:
            raise ValueError('Invalid spec: spec must contain a "model" key')
        if spec['model'] == 'linear':
            module_name = 'gemz.models.linear'
        else:
            raise NotImplementedError(f'No such model: {spec["model"]}')

        module = importlib.import_module(module_name)

        return module.make_model(spec, None)

    @classmethod
    def from_conditionner(cls, spec: ModelSpec, unobserved: Conditioner):
        """
        Instantiate a model from a specification document and a model conditioner
        """
        return cls(spec, unobserved)

    @property
    def conditional(self):
        """
        Return an indexable object for convenient specification of conditional
        tasks
        """
        return ConditionMaker(self)

    def mean(self, *data, **kwdata):
        """
        Mean of unobserved data given observed data and parameters
        """
        raise NotImplementedError

@dataclass
class ConditionMaker:
    """
    Indexable object to easily condition a model
    """
    model: Model

    def __getitem__(self, keys):
        """
        Returns a conditionned copy of the original model
        """
        return self.model.from_conditionner(self.model,
                Conditioner.from_indexes(keys))
