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
    mean: dict | None = None

    def observe(self, observed: dict) -> 'MatrixTObservation':
        """
        Make observation from data using model specified by self
        """
        return MatrixTObservation(**dc.asdict(self), observed=observed)

    def __post_init__(self):
        self.len_left = sum(self.left_dims.values())
        self.len_right = sum(self.right_dims.values())

    def as_dense(self) -> mt.MatrixT:
        """
        Pack all arguments into a dense matrix-t
        """
        return mt.MatrixT(
                self.dfs,
                dok_to_dense(self.left_dims, self.left_dims, self.left),
                dok_to_dense(self.right_dims, self.right_dims, self.right),
                )

@dataclass
class MatrixTObservation(MatrixT):
    """
    Observation from the parent matrix-t distribution
    """
    observed: dict

    def generator(self) -> dict:
        """
        Generator matrix, as a DoK
        """
        if self.mean is None:
            cobs = self.observed
        else:
            cobs = dok_sub(self.observed, self.mean)
        mcobs_t = dok_neg(dok_transpose(cobs))
        return self.left | self.right | cobs | mcobs_t

    def log_pdf(self):
        """
        Log-density
        """
        _sign, logdet_left = dok_slogdet(self.left_dims, self.left)
        _sign, logdet_right = dok_slogdet(self.right_dims, self.right)

        _sign, logdet = dok_slogdet(self.left_dims | self.right_dims, self.generator())

        return (
                0.5 * (self.dfs + self.len_right - 1) * logdet_right
                + 0.5 * (self.dfs + self.len_left - 1) * logdet_left
                - mt.log_norm_std(self.dfs, self.len_left, self.len_right)
                - 0.5 * (self.dfs + self.len_left + self.len_right - 1) * logdet
                )

    def post_left(self) -> 'Wishart':
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

    def as_dense(self) -> mt.MatrixTObservation:
        """
        Pack into dense mto
        """
        mtd = super().as_dense()
        return mt.observe(mtd, dok_to_dense(
            self.left_dims, self.right_dims,
            self.observed
            ))

    def uni_cond(self) -> tuple:
        """
        One-dimensional conditionals
        """
        dense_mto = self.as_dense()
        dense_stats = mt.ref_uni_cond(dense_mto)

        return tuple(
            dense_to_dok(self.left_dims, self.right_dims, stat)
            for stat in dense_stats
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

@dataclass
class Wishart:
    """
    Specification of a block-Wishart distribution
    """
    dfs: float
    dims: dict[Any, int]
    gram: dict

    def __post_init__(self) -> None:
        self.len = sum(self.dims.values())

    def extend_right(self, right_dims: dict[Any, int], right: dict,
                     mean: dict | None = None) -> MatrixT:
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

@dataclass
class NonCentralMatrixT:
    """
    Helper to represent a matrix-t with a latent mean as an other matrix-t
    """
    mtd: MatrixT

    @classmethod
    def from_params(cls, dfs: float, left, right, gram_mean_left: float,
                    gram_mean_right: float) -> 'NonCentralMatrixT':

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

        return cls(MatrixT(
                dfs=dfs,
                left_dims=left_dims,
                right_dims=right_dims,
                left=left,
                right=right
                ))

    def observe(self, observed) -> 'NonCentralMatrixTObservation':
        """
        Add observed data to distribution.

        This adds lift variables as needed.
        """
        _observed = { ('left', 'right'): observed }

        if 'left_lift' in self.mtd.left_dims:
            _observed['left_lift', 'right'] = jnp.ones((
                1, observed.shape[-1]
                ))
        if 'right_lift' in self.mtd.right_dims:
            _observed['left', 'right_lift'] = jnp.ones((
                observed.shape[-2], 1
                ))

        # right_lift, left_lift implicitly set as zero
        mto = self.mtd.observe(_observed)
        return NonCentralMatrixTObservation(mto)

    @classmethod
    def from_left(cls, wishart_left: Wishart, right):
        """
        Alternate constructor to directly instantiate from a block-wishart
        already containing lift dimensions

        In this case, no extra lift dimensions needs to be added, as they will
        either be already present in the block-wishart and inherited from it, or
        have been absorbed (contracted over) during the conditioning operation
        that created the block-wishart.
        """
        right_dims = {'right': right.shape[-1]}
        right = { ('right', 'right'): right }
        mtd = wishart_left.extend_right(right_dims, right)
        return cls(mtd)

@dataclass
class NonCentralMatrixTObservation:
    """
    Non-central matrix-t observation.

    Wrapper around a representation as a central matrix-t with lift dimensions
    """
    mto: MatrixTObservation

    def post_left(self) -> 'NCWishart':
        """
        Compute the posterior distribution of the (block) left gram matrix after observing
        (block) data
        """
        return NCWishart(self.mto.post_left())

    def log_pdf(self):
        """
        Log density function
        """
        print('FIXME: this is a dummy')
        return self.mto.log_pdf()

    def uni_cond(self) -> tuple:
        """
        One-dimensional conditionals
        """
        all_stats = self.mto.uni_cond()

        return tuple(
            stat['left', 'right']
            for stat in all_stats
            )


@dataclass
class NCWishart:
    """
    Wrapper around the block-wishart representation of a conditionned noncentral
    matrix-t
    """
    wishart: Wishart

    def extend_right(self, right) -> NonCentralMatrixT:
        """
        Generate a matrix-t from an existing posterior left gram
        """
        return NonCentralMatrixT.from_left(self.wishart, right)
