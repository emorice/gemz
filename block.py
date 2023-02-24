"""
Block matrices utils
"""

import operator

from dataclasses import dataclass
from functools import partial
from typing import Any, TypeGuard
from collections import defaultdict
from collections.abc import KeysView, Mapping, Hashable
import itertools

import numpy as np
import jax.numpy as jnp

# Dimension types
# ================
# Generalize a shape (tuple of ints) to a tuples of nested dicts of ints,
# representing a possibly recursive "block shape"

NamedDim = Mapping[Hashable, 'Dim']
Dim = NamedDim | int
Dims = tuple[Dim, ...]
NamedDims = tuple[NamedDim, ...]

# Abstract array glue
# ===================
# We need to deal with collections of array-like objects of different types,
# especially our meta-arrays and jax arrays. They have close but distinct apis
# so some intermediate layer is necessary, provided by the classes below

class ArrayAPI:
    """
    Abstract interface to a library implementing array-like objects
    """
    @classmethod
    def array_function(cls, func, *args, **kwargs):
        """
        Maps a numpy function to an equivalent and apply it
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def zeros(cls, shape):
        """
        Create an array of specified shape full of zeros
        """
        raise NotImplementedError('Abstract method')


class JaxAPI(ArrayAPI):
    """
    Conversion layer from numpy to jax operations
    """
    functions = {
        np.diagonal: jnp.diagonal,
        np.outer: jnp.outer,
        np.linalg.inv: jnp.linalg.inv,
        np.linalg.slogdet: jnp.linalg.slogdet,
        }

    @classmethod
    def array_function(cls, func, *args, **kwargs):
        """
        Maps a numpy function to an equivalent and apply it
        """
        if func in cls.functions:
            return cls.functions[func](*args, **kwargs)
        raise NotImplementedError(func)

    @classmethod
    def zeros(cls, shape):
        return jnp.zeros(shape)

class MetaJaxAPI(JaxAPI):
    """
    Dymamic conversion to jax that tries to use numpy protocol first

    Array creation defaults to jax types.
    """
    @classmethod
    def array_function(cls, func, *args, **kwargs):
        """
        Apply function as-is if first positional argument defines
        an __array_function__, else try to use jax equivalent.
        """
        if args and hasattr(args[0], '__array_function__'):
            return func(*args, **kwargs)
        return super().array_function(func, *args, **kwargs)

# Core Block NDArray implementation
# ==================================
# Valid only with an ArrayAPI implementation, so you typically want to use a
# subclass instead

@dataclass
class BlockMatrix:
    """
    Wrapper around a dict of array-like objects representing a block matrix
    """

    dims: NamedDims
    blocks: dict[tuple, Any]

    aa = ArrayAPI

    # Block-wise operators
    # --------------------

    def __add__(self, other):
        return _blockwise_binop(operator.add, self, other)

    def __sub__(self, other):
        return _blockwise_binop(operator.sub, self, other)

    def __truediv__(self, other):
        return _blockwise_binop(operator.truediv, self, other)

    def __rmul__(self, other):
        return _blockwise_binop(operator.mul, other, self)

    def __pow__(self, other):
        return _blockwise_binop(operator.pow, self, other)

    # Other operators
    # ---------------
    def __neg__(self):
        return self.__class__(
                self.dims,
                dok_map(operator.neg, self.blocks)
                )

    def __matmul__(self, other):
        return self.__class__(
                self.dims,
                dok_product(self.blocks, other.blocks)
                )

    # Numpy protocol
    # --------------

    def __array_function__(self, func, types, args, kwargs):
        if func is np.linalg.inv:
            return self._inv()

        if func is np.linalg.slogdet:
            return self._slogdet()

        if func is np.diagonal:
            return self.diagonal()

        if func is np.outer:
            if kwargs or len(args) != 2 or args[0] is not self:
                return NotImplemented
            return self._outer(args[1])

        return NotImplemented

    def __array_ufunc__(self, ufunc, name, *args, **kwargs):
        if name == '__call__':
            if len(args) == 1 and args[0] is self and not kwargs:
                jax_ufunc = getattr(jnp, ufunc.__name__)
                return self.__class__(
                        self.dims,
                        dok_map(jax_ufunc, self.blocks)
                        )
        return NotImplemented

    # Core linalg implementations
    # ---------------------------

    def _ensure_square(self, axis1=-2, axis2=-1):
        """
        Raise if not square
        """
        if len(self.dims) >= 2:
            if self.dims[axis1] == self.dims[axis2]:
                return
        raise RuntimeError('Matrix is not square')

    def _inv(self):
        """
        Fallback inversion using dense matrices
        """
        self._ensure_square()
        return self.from_dense(
                self.dims,
                self.aa.array_function(
                    np.linalg.inv, self.to_dense()
                    )
                )

    def _slogdet(self):
        """
        Fallback slogdet using dense matrices
        """
        self._ensure_square()
        return self.aa.array_function(
            np.linalg.slogdet, self.to_dense()
            )

    def _outer(self, other):
        """
        Technically not consistent with np.outer as this does not flatten
        arguments
        """
        blocks = {}
        for skey, svalue in self.blocks.items():
            for okey, ovalue in other.blocks.items():
                new = self.aa.array_function(np.outer, svalue, ovalue)
                blocks[tuple((*skey, *okey))] = new
        return self.__class__.from_blocks(blocks)

    def diagonal(self, axis1=0, axis2=1):
        """
        Diagonal of square array
        """
        self._ensure_square(axis1, axis2)

        blocks = {}
        for key, value in self.blocks.items():
            if key[axis1] == key[axis2]:
                new_key = tuple(ki for i, ki in enumerate(key)
                                if i not in (axis1, axis2))
                new_key = (*new_key, key[axis1])
                blocks[new_key] = self.aa.array_function(np.diagonal, value)
        return self.__class__.from_blocks(blocks)

    @property
    def T(self): # pylint: disable=invalid-name
        """
        Transpose
        """
        return self.__class__(
                self.dims[::-1],
                {
                    key[::-1]: value.T
                    for key, value in self.blocks.items()
                    }
                )
    @property
    def shape(self) -> tuple[int, ...]:
        """
        Total shape, summing over blocks shapes
        """
        return dims_to_shape(self.dims)

    # Creation and conversion methods
    # -------------------------------

    @classmethod
    def indexes(cls, array):
        """
        Unified interface to generalized shape
        """
        if isinstance(array, BlockMatrix):
            return array.dims
        return np.shape(array)

    def to_dense(self):
        """
        Convert self to dense matrix, replacing missing blocks with dense zeros
        """
        result = self.aa.zeros(self.shape)
        cumshape = dims_to_cumshape(self.dims)

        for key, block in self.blocks.items():
            indexer = _key_to_indexer(key, self.dims, cumshape)
            dense_block = self.as_dense(block)
            # Fixme: this only works with jax outputs
            result = result.at[indexer].set(dense_block)

        return result

    @classmethod
    def as_dense(cls, array):
        """
        Convert array to dense if necesary
        """
        if isinstance(array, BlockMatrix):
            return array.to_dense()
        return array

    @classmethod
    def from_dense(cls, dims: Dims, dense):
        """
        Split array in blocks conforming to dims
        """
        if is_named(dims):
            cumshape = dims_to_cumshape(dims)
            blocks: dict = {}

            for key in itertools.product(*dims):
                indexer = _key_to_indexer(key, dims, cumshape)
                lower_dims = tuple(dim[ki] for dim, ki in zip(dims, key))
                blocks[key] = cls.from_dense(lower_dims, dense[indexer])

            return cls(dims, blocks)
        return dense

    @classmethod
    def from_blocks(cls, blocks: dict):
        """
        Build matrix from dictionnary of blocks, inferring dimensions
        """
        dims : dict[int, dict] = defaultdict(dict)
        for key, value in blocks.items():
            for idim, (key_item, length) in enumerate(
                    zip(key, cls.indexes(value))
                    ):
                dims[idim][key_item] = length
        return cls(tuple(dims.values()), blocks)

    @classmethod
    def zero(cls):
        """
        Empty block matrix, logically treated as zero
        """
        return cls.from_blocks({})

    @classmethod
    def zero2d(cls):
        """
        Empty block matrix, logically treated as zero, but still considered
        2-dimensional
        """
        return cls(({}, {}), {})

    def clone(self):
        """
        Copy of self. The blocks are not copied.
        """
        return self.__class__(
                tuple(dict(dim) for dim in self.dims),
                dict(self.blocks)
                )

    # Indexing methods
    # ----------------

    def __setitem__(self, key, value):
        for idim, (key_item, length) in enumerate(
                zip(key, np.shape(value))
                ):
            self.dims[idim][key_item] = length
        self.blocks[key] = value

    def __getitem__(self, key):
        # Indexing a 1d array with a set, normalize to a 1-tuple
        if self._is_multikey(key):
            key = (key,)

        # At this point key is a dim-matching tuple, except simple key for 1-d
        # case

        if isinstance(key, tuple) and any(self._is_multikey(ki) for ki in key):
            key = tuple(self._as_multikey(ki, axis=i)
                    for i, ki in enumerate(key))
            blocks = {}
            for _key, block in self.blocks.items():
                if all(ki in kset for ki, kset in zip(_key, key)):
                    blocks[_key] = block
            return self.__class__.from_blocks(blocks)

        # In the 1-d case, our keys are 1-tuples so basic keys needs to be
        # wrapped to make array[key] behave as  array[key,]
        if not isinstance(key, tuple):
            key = (key,)
        return self.blocks[key]

    def _is_multikey(self, key):
        return isinstance(key, (set, list, slice, KeysView))

    def _as_multikey(self, key, axis):
        if key == slice(None):
            return self.dims[axis].keys()
        if self._is_multikey(key):
            return key
        return {key}

# Concrete subclasses
# ===================

class JaxBlockMatrix(BlockMatrix):
    """
    Block matrix using jax array as its base type
    """
    aa = MetaJaxAPI

# Helper functions
# ================

def _blockwise_binop(binop, left, right):
    """
    Generic implementation of binary operators that act block per block
    """
    if isinstance(left, BlockMatrix):
        if isinstance(right, BlockMatrix):
            # Two block matrices, do a binary map
            # Class and dims arbitrarily taken from left arg
            # Missing blocks treated as as scalar zeros
            return left.__class__(
                left.dims,
                dok_map(binop, left.blocks, right.blocks, fill=0.)
                )
        # Left is block but right is not. Try to broadcast right against all
        # blocks
        return left.__class__(
            left.dims,
            dok_map(lambda b: binop(b, right), left.blocks)
            )
    if isinstance(right, BlockMatrix):
        # Right is block but left is not. Try to broadcast left against all
        # blocks of right
        return right.__class__(
            right.dims,
            dok_map(lambda b: binop(left, b), right.blocks)
            )
    # None are blocks
    raise NotImplementedError

# Dimension helpers

def is_named(dims: Dims) -> TypeGuard[NamedDims]:
    """
    Check if a Dims object is actually a NamedDims, that this a tuple of
    dictionnaries and not a tuple of ints
    """
    return all(isinstance(dim, Mapping) for dim in dims)

def dim_len(dim: Dim) -> int:
    """
    Total length along a dimension (i.e. recursive sum of lengths)
    """
    if isinstance(dim, Mapping):
        return sum(map(dim_len, dim.values()))
    return dim

def dims_to_shape(dims: Dims) -> tuple[int, ...]:
    """
    Numeric shape of a tuple of dims.
    """
    return tuple(map(dim_len, dims))

def dims_to_cumshape(dims: NamedDims) -> tuple[dict[Any, int], ...]:
    """
    Cumulative length of dimension items.

    Effectively gives the index in the dense matrix corresponding to a block
    """
    cumshape: list = []
    for dim in dims:
        length = 0
        cumlen: dict = {}
        for key, dim_item in dim.items():
            cumlen[key] = length
            length += dim_len(dim_item)
        cumshape.append(cumlen)
    return tuple(cumshape)

def _key_to_indexer(key, dims, cumshape):
    return tuple(
            slice(
                cumshape[idim][ki],
                cumshape[idim][ki] + dim_len(dims[idim][ki])
                )
            for idim, ki in enumerate(key)
            )

# Dictionnary of block operations helpers
# ========================================

def dok_product(left_dok: dict, right_dok: dict) -> dict:
    """
    Matrix product of doks.

    Missing blocks efficiently treated as zeros.

    Inefficient for large number of small blocks (quadratic).
    """
    result: dict = {}

    for (left_left, left_right), left_value in left_dok.items():
        for (right_left, right_right), right_value in right_dok.items():
            if left_right != right_left:
                continue
            if (left_left, right_right) in result:
                result[left_left, right_right] += left_value @ right_value
            else:
                result[left_left, right_right] = left_value @ right_value
    return result

def dok_map(function, *doks: dict, fill=None):
    """
    Apply function element-wise to DoKs

    Since DoKs may not have the same keys, a filler element can be used on
    non-matching keys.
    """
    all_keys = { keys for dok in doks for keys in dok }

    return {
        keys: function(*(
            dok[keys] if keys in dok else fill
            for dok in doks
            ))
        for keys in all_keys
        }
