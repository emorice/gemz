"""
Block-wise description of matrix-t variates
"""

import dataclasses as dc

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from block import BlockMatrix

import matrix_t as mt

@dataclass
class MatrixT:
    """
    Specification of a matrix-t distribution
    """
    dfs: float
    left: BlockMatrix
    right: BlockMatrix

    _: dc.KW_ONLY
    mean: BlockMatrix | None = None

    def observe(self, observed: BlockMatrix) -> 'MatrixTObservation':
        """
        Make observation from data using model specified by self
        """
        return MatrixTObservation(self, observed=observed)

    def __post_init__(self):
        self.len_left = self.left.shape[-1]
        self.len_right = self.right.shape[-1]

    def as_dense(self) -> mt.MatrixT:
        """
        Pack all arguments into a dense matrix-t
        """
        return mt.MatrixT(
                self.dfs,
                self.left.as_dense(),
                self.right.as_dense()
                )

@dataclass
class MatrixTObservation:
    """
    Observation from the parent matrix-t distribution
    """
    mtd: MatrixT
    observed: BlockMatrix

    def generator(self) -> BlockMatrix:
        """
        Generator matrix, as a DoK
        """
        if self.mtd.mean is None:
            cobs = self.observed
        else:
            cobs = self.observed - self.mtd.mean
        mcobs_t = - cobs.T
        return self.mtd.left | self.mtd.right | cobs | mcobs_t

    def log_pdf(self):
        """
        Log-density
        """
        _sign, logdet_left = np.linalg.slogdet(self.mtd.left)
        _sign, logdet_right = np.linalg.slogdet(self.mtd.right)

        _sign, logdet = np.linalg.slogdet(self.generator())

        return (
                0.5 * (self.mtd.dfs + self.mtd.len_right - 1) * logdet_right
                + 0.5 * (self.mtd.dfs + self.mtd.len_left - 1) * logdet_left
                - mt.log_norm_std(self.mtd.dfs, self.mtd.len_left, self.mtd.len_right)
                - 0.5 * (self.mtd.dfs + self.mtd.len_left + self.mtd.len_right - 1) * logdet
                )

    def post_left(self) -> 'Wishart':
        """
        Compute the posterior distribution of the (block) left gram matrix after observing
        (block) data
        """
        return Wishart(
            dfs=self.mtd.dfs + self.mtd.len_right,
            gram=self.mtd.left + self.observed @ (
                np.linalg.inv(self.mtd.right) @ self.observed.T
                )
            )

    def as_dense(self) -> mt.MatrixTObservation:
        """
        Pack into dense mto
        """
        mtd = self.mtd.as_dense()
        return mt.observe(mtd, self.observed.as_dense())

    def uni_cond(self) -> tuple:
        """
        One-dimensional conditionals
        """
        dense_mto = self.as_dense()
        dense_stats = mt.ref_uni_cond(dense_mto)

        return tuple(
            BlockMatrix.from_dense(self.observed.left_dims, self.observed.right_dims, stat)
            for stat in dense_stats
            )

@dataclass
class Wishart:
    """
    Specification of a block-Wishart distribution
    """
    dfs: float
    gram: BlockMatrix

    def __post_init__(self) -> None:
        self.len = self.gram.shape[-1]

    def extend_right(self, right: BlockMatrix, mean: BlockMatrix | None = None) -> MatrixT:
        """
        Generate a matrix-t from an existing posterior left gram
        """
        return MatrixT(
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

        left = { ('left', 'left'): left }
        right = { ('right', 'right'): right }

        if gram_mean_left is not None:
            # Augment left with lift dimension
            left['left_lift', 'left_lift'] = gram_mean_left * jnp.eye(1)

        if gram_mean_right is not None:
            # Augment right with lift dimension
            right['right_lift', 'right_lift'] = gram_mean_right * jnp.eye(1)

        return cls(MatrixT(
                dfs=dfs,
                left=BlockMatrix.from_blocks(left),
                right=BlockMatrix.from_blocks(right),
                ))

    def observe(self, observed) -> 'NonCentralMatrixTObservation':
        """
        Add observed data to distribution.

        This adds lift variables as needed.
        """
        _observed = { ('left', 'right'): observed }

        if 'left_lift' in self.mtd.left.left_dims:
            _observed['left_lift', 'right'] = jnp.ones((
                1, observed.shape[-1]
                ))
        if 'right_lift' in self.mtd.right.right_dims:
            _observed['left', 'right_lift'] = jnp.ones((
                observed.shape[-2], 1
                ))

        # right_lift, left_lift implicitly set as zero
        mto = self.mtd.observe(BlockMatrix.from_blocks(_observed))
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
        right = { ('right', 'right'): right }
        mtd = wishart_left.extend_right(BlockMatrix.from_blocks(right))
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
