"""
Block-wise description of matrix-t variates
"""

import dataclasses as dc
import operator

from dataclasses import dataclass
from typing import Any
from functools import partial

import jax.numpy as jnp

import matrix_t as mt

@dataclass
class MatrixT:
    """
    Specification of a matrix-t distribution
    """
    left_dims: dict[Any, int]
    right_dims: dict[Any, int]

    dfs: float
    left: dict
    right: dict

    _: dc.KW_ONLY
    mean: dict

    def observe(self, observed: dict):
        """
        Make observation from data using model specified by self
        """
        return MTO(**dc.asdict(self), observed=observed)

    def __post_init__(self):
        self.len_left = sum(self.left_dims.values())
        self.len_right = sum(self.right_dims.values())

@dataclass
class MTO(MatrixT):
    """
    Observation from the parent matrix-t distribution
    """
    observed: dict

    def generator(self):
        """
        Generator matrix, as a DoK
        """
        cobs = dok_map(operator.sub, self.observed, self.mean)
        mcobs_t = dok_map(operator.neg, dok_transpose(cobs))
        return self.left | self.right | cobs | mcobs_t

    def log_pdf(self):
        """
        Log-density
        """
        _sign, logdet_left = dok_slogdet(self.left)
        _sign, logdet_right = dok_slogdet(self.right)

        _sign, logdet = dok_slogdet(self.generator())

        return (
                0.5 * (self.dfs + self.len_right - 1) * logdet_right
                + 0.5 * (self.dfs + self.len_left - 1) * logdet_left
                - mt.log_norm_std(self.dfs, self.len_left, self.len_right)
                - 0.5 * (self.dfs + self.len_left + self.len_right - 1) * logdet
                )

    def post_left(self) -> Wishart:
        """
        Compute the posterior distribution of the (block) left gram matrix after observing
        (block) data
        """
        return Wishart(
            dfs=self.dfs + self.len_right,
            dims=self.left_dims,
            gram=dok_add(self.left, dok_product(self.observed,
                dok_product(dok_inv(self.right_dims, self.right), dok_transpose(self.observed))
                ))
            )

def dok_to_lol(left_dims: dict, right_dims: dict, dok: dict[Any, dict]
               ) -> list[list]:
    """
    Convert dict-of-dict to list-of-list with the given orderings
    """
    return [
            [
                dok[ikey, jkey]
                for jkey in right_dims
            ]
            for ikey in left_dims
            ]

def dok_to_dense(*args, **kwargs):
    """
    Convert dict-of-dict to dense matrix
    """
    return jnp.block(dok_to_lol(*args, **kwargs))

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
            dok_to_dense(dok)
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

def dok_map(function, *doks):
    """
    Apply function element-wise to DoKs
    """
    return {
        keys: function(*(
            dok[keys]
            for dok in doks
            ))
        for keys in doks[0]
        }

def dok_transpose(dok: dict):
    """
    Transpose a DoK
    """
    return {
        (right_key, left_key): value.T
        for (left_key, right_key), value in dok.items()
        }

def dok_slogdet(dok: dict[Any, dict]):
    """
    Naive determinant of dict-of-dict block matrix
    """
    return  jnp.linalg.slogdet(
        dok_to_dense(dok)
        )

dok_add = partial(dok_map, operator.add)
dok_sub = partial(dok_map, operator.sub)

@dataclass
class Wishart:
    """
    Specification of a block-Wishart distribution
    """
    dfs: float
    dims: dict[Any, int]
    gram: dict

    def __post_init__(self):
        self.len = sum(self.dims.values())

    def extend_right(self, right_dims: dict[Any, int], right: dict, mean=0.) -> MatrixT:
        """
        Generate a matrix-t from an existing posterior left gram
        """
        return MatrixT(
                left_dims=self.dims,
                right_dims=right_dims,
                dfs=self.dfs,
                left=self.gram,
                right=right,
                mean=mean
                )

class NonCentralMatrixT:
    """
    Helper to represent a matrix-t with a latent mean as an other matrix-t
    """
    def __init__(self, dfs, left, right, gram_mean_left, gram_mean_right):

        left_dims = {'left': left.shape[-1]}
        left = { ('left', 'left'): left }

        right_dims = {'right': right.shape[-1]}
        right = { ('right', 'right'): right }

        if gram_mean_left is not None:
            # Augment left with lift dimension
            left_dims['left_lift'] = 1
            left['left_lift', 'left_lift'] = gram_mean_left * jnp.eye(1)

        if gram_mean_right is not None:
            # Augment right with lift dimension
            right_dims['right_lift'] = 1
            right['right_lift', 'right_lift'] = gram_mean_right * jnp.eye(1)

        self.mtd = MatrixT(
                dfs=dfs,
                left_dims=left_dims,
                right_dims=right_dims,
                left=left,
                right=right
                )

    def observe(self, observed):
        """
        Add observed data to distribution.

        This adds lift variables as needed.
        """

        return NCMTO(self.mtd, observed)

class NCMTO:
    """
    Non-central matrix-t observation.

    Wrapper around a representation as a central matrix-t with lift dimensions
    """
    def __init__(self, ncmtd: NonCentralMatrixT, observed):
        mtd = ncmtd.mtd

        _observed = { ('left', 'right'): observed }

        if 'left_lift' in mtd.left_dims:
            _observed['left_lift', 'right'] = jnp.ones_like(observed[0])
        if 'right_lift' in mtd.right_dims:
            _observed['left', 'right_lift'] = jnp.ones_like(observed[0, :])

        # right_lift, left_lift implicitly set as zero
        self.mto = MTO(**dc.asdict(mtd), observed=_observed)
