"""
Block matrices utils
"""

import operator

from dataclasses import dataclass
from functools import partial
from typing import Any
from collections import defaultdict
from collections.abc import KeysView

import numpy as np
import jax.numpy as jnp

@dataclass
class BlockMatrix:
    """
    Wrapper around a dict of array-like objects representing a block matrix
    """

    dims: tuple[dict, ...]
    blocks: dict

    def __add__(self, other):
        return self.__class__(
                self.dims,
                dok_add(self.blocks, other.blocks)
                )

    def __sub__(self, other):
        return self.__class__(
                self.dims,
                dok_sub(self.blocks, other.blocks)
                )

    def __neg__(self):
        return self.__class__(
                self.dims,
                dok_neg(self.blocks)
                )

    def __matmul__(self, other):
        return self.__class__(
                self.dims,
                dok_product(self.blocks, other.blocks)
                )

    def __array_function__(self, func, types, args, kwargs):
        if func is np.linalg.inv:
            self._ensure_square()
            return self.__class__(
                    self.dims,
                    dok_inv(self.dims[-1], self.blocks)
                    )

        if func is np.linalg.slogdet:
            self._ensure_square()
            return dok_slogdet(self.dims[-1], self.blocks)

        if func is np.diagonal:
            return self.diagonal()

        return NotImplemented

    def _ensure_square(self, axis1=-2, axis2=-1):
        """
        Raise if not square
        """
        if len(self.dims) >= 2:
            if self.dims[axis1] == self.dims[axis2]:
                return
        raise RuntimeError('Matrix is not square')

    def diagonal(self, axis1=0, axis2=1):
        """
        Diagonal of square array
        """
        # Note: the numpy convention is unfortunate here, last and
        # second-to-last axes as default would be more consistent. Sticking with
        # numpy api nethertheless for least surprise.
        # Resulting axis is appended at the end, which by itself is sound but
        # makes the defaults even weirder.
        self._ensure_square(axis1, axis2)

        blocks = {}
        for key, value in self.blocks.items():
            if key[axis1] == key[axis2]:
                new_key = tuple(ki for i, ki in enumerate(key)
                                if i not in (axis1, axis2))
                new_key = (*new_key, key[axis1])
                blocks[new_key] = value
        return self.__class__.from_blocks(blocks)

    @property
    def T(self): # pylint: disable=invalid-name
        """
        Transpose
        """
        return self.__class__(
                self.dims[::-1],
                dok_transpose(self.blocks)
                )

    def as_dense(self):
        """
        Concatenate all blocks into a dense matrix
        """
        return dok_to_dense(self.dims, self.blocks)

    @classmethod
    def from_dense(cls, dims: tuple, dense):
        """
        Split dense matrix into blocks
        """
        return cls(
                dims,
                dense_to_dok(dims, dense)
                )

    @classmethod
    def from_blocks(cls, blocks: dict):
        """
        Build matrix from dictionnary of blocks, inferring dimensions
        """
        dims : dict[int, dict] = defaultdict(dict)
        for key, value in blocks.items():
            for idim, (key_item, length) in enumerate(
                    zip(key, np.shape(value))
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

        # Todo: normalize simple 1-tuples
        return self.blocks[key]

    def _is_multikey(self, key):
        return isinstance(key, (set, list, slice, KeysView))

    def _as_multikey(self, key, axis):
        if key == slice(None):
            return self.dims[axis].keys()
        if self._is_multikey(key):
            return key
        return {key}

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Total shape, summing over blocks shapes
        """
        return tuple(sum(dim.values()) for dim in self.dims)

    def __or__(self, other):
        return self.__class__(
                tuple(sdim | odim for (sdim, odim) in
                      zip(self.dims, other.dims)),
                self.blocks | other.blocks
                )

    def clone(self):
        """
        Copy of self. The blocks are not copied.
        """
        return self.__class__(
                tuple(dict(dim) for dim in self.dims),
                dict(self.blocks)
                )

def dok_to_lol(dims: tuple[dict, ...], dok: dict) -> list:
    """
    Convert dict of blocks to nested lists with the given orderings
    If entries are missing, they are replaced with dense 0 matrices of the
    correct dimension.
    """
    if len(dims) != 2:
        raise NotImplementedError
    return [
            [
                dok[ikey, jkey]
                if (ikey, jkey) in dok
                else jnp.zeros((ilen, jlen))
                for jkey, jlen in dims[1].items()
            ]
            for ikey, ilen in dims[0].items()
            ]

def dok_to_dense(dims: tuple[dict, ...], dok: dict):
    """
    Convert dict-of-dict to dense matrix
    """
    return jnp.block(dok_to_lol(dims, dok))

def dense_to_dok(dims: tuple[dict[Any, int], ...], dense):
    """
    Split a dense matrix back into a DoK according to partitions
    """
    if len(dims) != 2:
        raise NotImplementedError
    dok = {}
    i = 0
    for left_key, left_len in dims[0].items():
        j = 0
        for right_key, right_len in dims[1].items():
            dok[left_key, right_key] = dense[i:i+left_len, j:j+right_len]
            j += right_len
        i += left_len
    return dok

def dok_inv(dim: dict[Any, int], dok: dict):
    """
    Naive DoK invert by dense intermediate
    """
    return dense_to_dok(
        (dim, dim),
        jnp.linalg.inv(
            dok_to_dense((dim, dim), dok)
            )
        )

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
            dok[keys]
            if keys in dok
            else fill
            for dok in doks
            ))
        for keys in all_keys
        }

def dok_transpose(dok: dict) -> dict:
    """
    Transpose a DoK
    """
    return {
            key[::-1]: value.T
        for key, value in dok.items()
        }

def dok_slogdet(dims: dict[Any, int], dok: dict[Any, dict]):
    """
    Naive determinant of dict-of-dict block matrix
    """
    return  jnp.linalg.slogdet(
        dok_to_dense((dims, dims), dok)
        )

dok_add = partial(dok_map, operator.add, fill=0.)
dok_sub = partial(dok_map, operator.sub, fill=0.)
dok_neg = partial(dok_map, operator.neg)
