"""
Block matrices utils
"""

import operator

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import jax.numpy as jnp

@dataclass
class BlockMatrix:
    """
    Wrapper around a dict of array-like objects representing a block matrix
    """

    left_dims: dict[Any, int]
    right_dims: dict[Any, int]
    blocks: dict

    def __add__(self, other):
        return self.__class__(
                self.left_dims, self.right_dims,
                dok_add(self.blocks, other.blocks)
                )

    def __sub__(self, other):
        return self.__class__(
                self.left_dims, self.right_dims,
                dok_sub(self.blocks, other.blocks)
                )

    def __neg__(self):
        return self.__class__(
                self.left_dims, self.right_dims,
                dok_neg(self.blocks)
                )

    def __matmul__(self, other):
        return self.__class__(
                self.left_dims, other.right_dims,
                dok_product(self.blocks, other.blocks)
                )

    def __array_function__(self, func, types, args, kwargs):
        if func is np.linalg.inv:
            if self.left_dims != self.right_dims:
                raise RuntimeError('Matrix is not square')
            return self.__class__(
                    self.left_dims, self.right_dims,
                    dok_inv(self.left_dims, self.blocks)
                    )

        if func is np.linalg.slogdet:
            if self.left_dims != self.right_dims:
                raise RuntimeError('Matrix is not square')
            return dok_slogdet(self.left_dims, self.blocks)

        return NotImplemented

    @property
    def T(self): # pylint: disable=invalid-name
        """
        Transpose
        """
        return self.__class__(
                self.right_dims, self.left_dims,
                dok_transpose(self.blocks)
                )

    def as_dense(self):
        """
        Concatenate all blocks into a dense matrix
        """
        return dok_to_dense(self.left_dims, self.right_dims, self.blocks)

    @classmethod
    def from_dense(cls, left_dims, right_dims, dense):
        """
        Split dense matrix into blocks
        """
        return cls(
                left_dims, right_dims,
                dense_to_dok(left_dims, right_dims, dense)
                )

    @classmethod
    def from_blocks(cls, blocks: dict):
        """
        Build matrix from dictionnary of blocks, inferring dimensions
        """
        left_dims = {}
        right_dims = {}
        for (left, right), value in blocks.items():
            left_dims[left], right_dims[right] = value.shape
        return cls(left_dims, right_dims, blocks)

    def __setitem__(self, key, value):
        left, right = key
        self.left_dims[left], self.right_dims[right] = value.shape
        self.blocks[key] = value

    def __getitem__(self, key):
        return self.blocks[key]

    @property
    def shape(self) -> tuple[int, int]:
        """
        Total shape, summing over blocks shapes
        """
        return (sum(self.left_dims.values()), sum(self.right_dims.values()))

    def __or__(self, other):
        return self.__class__(
                self.left_dims | other.left_dims,
                self.right_dims | other.right_dims,
                self.blocks | other.blocks
                )

def dok_to_lol(left_dims: dict, right_dims: dict, dok: dict[Any, dict]
               ) -> list[list]:
    """
    Convert dict-of-dict to list-of-list with the given orderings
    If entries are missing, they are replaced with dense 0 matrices of the
    correct dimension.
    """
    return [
            [
                dok[ikey, jkey]
                if (ikey, jkey) in dok
                else jnp.zeros((ilen, jlen))
                for jkey, jlen in right_dims.items()
            ]
            for ikey, ilen in left_dims.items()
            ]

def dok_to_dense(left_dims: dict, right_dims: dict, dok: dict[Any, dict]):
    """
    Convert dict-of-dict to dense matrix
    """
    return jnp.block(dok_to_lol(left_dims, right_dims, dok))

def dense_to_dok(left_dims: dict[Any, int], right_dims: dict[Any, int], dense):
    """
    Split a dense matrix back into a DoK according to partitions
    """
    dok = {}
    i = 0
    for left_key, left_len in left_dims.items():
        j = 0
        for right_key, right_len in right_dims.items():
            dok[left_key, right_key] = dense[i:i+left_len, j:j+right_len]
            j += right_len
        i += left_len
    return dok

def dok_inv(dims: dict[Any, int], dok: dict):
    """
    Naive DoK invert by dense intermediate
    """
    return dense_to_dok(
        dims, dims,
        jnp.linalg.inv(
            dok_to_dense(dims, dims, dok)
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

def dok_transpose(dok: dict):
    """
    Transpose a DoK
    """
    return {
        (right_key, left_key): value.T
        for (left_key, right_key), value in dok.items()
        }

def dok_slogdet(dims: dict[Any, int], dok: dict[Any, dict]):
    """
    Naive determinant of dict-of-dict block matrix
    """
    return  jnp.linalg.slogdet(
        dok_to_dense(dims, dims, dok)
        )

dok_add = partial(dok_map, operator.add, fill=0.)
dok_sub = partial(dok_map, operator.sub, fill=0.)
dok_neg = partial(dok_map, operator.neg)
