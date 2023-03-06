"""
Block-wise description of matrix-t variates
"""

import dataclasses as dc
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsc

from block_jax import JaxBlockMatrix as BlockMatrix

def log_norm_std(dfs, len_left, len_right):
    """
    Log normalization constant of a standard matrix-t
    """
    small = min(len_left, len_right)
    large = max(len_left, len_right)

    args = 0.5 * (dfs + jnp.arange(small))

    return (
        jnp.sum(jsc.gammaln(args))
        + 0.5 * small * large * jnp.log(jnp.pi)
        - jnp.sum(jsc.gammaln(args + 0.5*large))
        )

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

    def __post_init__(self):
        self.shape = (self.left.shape[-1], self.right.shape[-1])

    def condition_left(self, observed: BlockMatrix) -> 'MatrixT':
        """
        Condition on block rows of partial observations
        """
        keys = observed.dims[-2].keys()
        ckeys = self.left.dims[-1].keys() - keys

        centered = observed
        if self.mean is not None:
            centered = centered - self.mean[keys, :]

        pivot = self.left[keys, keys]
        inv_pivot = np.linalg.inv(pivot)
        trans = inv_pivot @ centered

        cond_left = self.left[ckeys, ckeys] - self.left[ckeys, keys] @ (
                inv_pivot @ self.left[keys, ckeys]
                )
        cond_right = self.right + np.swapaxes(centered, -1, -2) @ trans

        cond_mean = self.left[ckeys, keys] @ trans
        if self.mean is not None:
            cond_mean = self.mean[ckeys, :] + cond_mean
        cond_dfs = self.dfs + observed.shape[-2]

        return self.__class__(cond_dfs, cond_left, cond_right, mean=cond_mean)

    @property
    def grams(self):
        """
        Left and right gram matrices, but as a tuple for numeric indexing
        """
        return (self.left, self.right)

    def log_pdf(self, observed):
        """
        Log-density
        """
        log_inv_norm = (
                sum(
                    0.5 * (self.dfs + length - 1) * np.linalg.slogdet(gram)[1]
                    for length, gram in zip(self.shape, self.grams)
                    )
                - log_norm_std(self.dfs, *self.shape)
                )

        _sign, logdet = np.linalg.slogdet(self._generator(observed))
        return (
                log_inv_norm
                - 0.5 * (self.dfs + sum(self.shape) - 1) * logdet
                )

    def post(self, observed, axis: int = 0) -> 'Wishart':
        """
        Compute the posterior distribution of one of the gram matrix after observing
        data
        """
        other_axis = 1 - axis

        # obs with required axis first resp. last
        obs, obs_t = observed, np.swapaxes(observed, -1, -2)
        if axis == 1:
            obs, obs_t = obs_t, obs

        return Wishart(
            dfs=self.dfs + self.shape[other_axis],
            gram=self.grams[axis] + obs @ (
                np.linalg.inv(self.grams[other_axis]) @ obs_t
                )
            )

    def extend(self, observed, gram, axis: int = 0):
        """
        Shorthand for computing posterior and extending along axis
        """
        return self.post(observed, axis).extend(gram, axis)

    def _generator(self, observed) -> BlockMatrix:
        """
        Generator matrix, as a block matrix
        """
        cobs = observed
        if self.mean is not None:
            cobs = cobs - self.mean
        # Batched transpose
        cobs_t = np.swapaxes(cobs, -1, -2)
        return BlockMatrix.from_blocks({
            (0, 0): self.left,  (0, 1): cobs,
            (1, 0): - cobs_t,   (1, 1): self.right,
            })

    def uni_cond(self, observed) -> tuple[BlockMatrix, ...]:
        """
        Conditional distributions of individual entries
        """
        igmat = np.linalg.inv(self._generator(observed))

        inv_diag = np.diagonal(igmat)
        inv_diag_left, inv_diag_right = inv_diag[0], inv_diag[1]

        inv_data = igmat[0, 1]

        # Broken from here, wip
        # Note on the outer product: np.outer is a legacy function.
        # np.ufunc.outer is more general, but has no jax implementation.
        # Using broadcasting would probably be the most portable, but is more
        # work to implement
        inv_diag_prod = np.outer(inv_diag_left, inv_diag_right)

        dets = inv_diag_prod + inv_data**2

        residuals = - inv_data / dets

        dfs = self.dfs + sum(self.shape) - 2
        means = observed - residuals
        variances = inv_diag_prod / ((dfs - 2.) * dets**2)
        logks = 0.5 * (dfs * np.log(inv_diag_prod) - (dfs - 1) * np.log(dets))
        logps = logks - jsc.betaln(0.5 * dfs, 0.5)
        return means, variances, logps

@dataclass
class Wishart:
    """
    Specification of a block-Wishart distribution
    """
    dfs: float
    gram: BlockMatrix

    def __post_init__(self) -> None:
        self.len = self.gram.shape[-1]

    def extend(self, new_gram: BlockMatrix, axis: int = 0, mean: BlockMatrix | None = None) -> MatrixT:
        """
        Generate a matrix-t from an existing gram posterior, using it as the
        axis side
        """
        return MatrixT(
                dfs=self.dfs,
                left=self.gram if axis == 0 else new_gram,
                right=self.gram if axis == 1 else new_gram,
                mean=mean,
                )

@dataclass
class NonCentralMatrixT:
    """
    Helper to represent a matrix-t with a latent mean as an other matrix-t
    """
    mtd: MatrixT

    @classmethod
    def from_params(cls, dfs: float, left, right,
            gram_mean_left: float | None = None,
            gram_mean_right: float | None = None,
            ) -> 'NonCentralMatrixT':

        left = { ('obs', 'obs'): left }
        right = { ('obs', 'obs'): right }

        if gram_mean_left is not None:
            # Augment left with lift dimension
            left['lift', 'lift'] = gram_mean_left * jnp.eye(1)

        if gram_mean_right is not None:
            # Augment right with lift dimension
            right['lift', 'lift'] = gram_mean_right * jnp.eye(1)

        return cls(MatrixT(
                dfs=dfs,
                left=BlockMatrix.from_blocks(left),
                right=BlockMatrix.from_blocks(right),
                ))

    @classmethod
    def from_post(cls, wishart: Wishart, new_gram, axis: int = 0) -> 'NonCentralMatrixT':
        """
        Alternate constructor to directly instantiate from a block-wishart
        already containing lift dimensions

        In this case, no extra lift dimensions needs to be added, as they will
        either be already present in the block-wishart and inherited from it, or
        have been absorbed (contracted over) during the conditioning operation
        that created the block-wishart.
        """
        _new_gram = { ('obs', 'obs'): new_gram }
        mtd = wishart.extend(BlockMatrix.from_blocks(_new_gram), axis)
        return cls(mtd)

    def condition_on_lift(self) -> MatrixT:
        """
        Condition on lift variables
        """
        if 'lift' in self.mtd.right.dims[-1]:
            raise NotImplementedError
        if 'lift' in self.mtd.left.dims[-1]:
            mtd = self.mtd.condition_left(BlockMatrix.from_blocks({
                ('lift', 'obs'): jnp.ones((
                    1, self.mtd.right.dims[-1]['obs']
                    ))
                }))
        else:
            mtd = self.mtd
        return mtd

    def log_pdf(self, observed):
        """
        Log density function
        """
        return self.condition_on_lift().log_pdf(self.wrap_observed(observed))

    def wrap_observed(self, observed):
        """
        Wrap observed in a block matrix
        """
        return BlockMatrix.from_blocks(
                { ('obs', 'obs'): observed }
                )

    def augment_observed(self, observed):
        """
        Pad observation with lift dimensions
        """
        _blocks = { ('obs', 'obs'): observed }

        if 'lift' in self.mtd.left.dims[-1]:
            _blocks['lift', 'obs'] = jnp.ones((
                1, self.mtd.right.dims[-1]['obs']
                ))
        if 'lift' in self.mtd.right.dims[-1]:
            _blocks['obs', 'lift'] = jnp.ones((
                self.mtd.left.dims[-1]['obs'], 1
                ))
        return BlockMatrix.from_blocks(_blocks)

    def post(self, observed, axis: int = 0) -> 'NCWishart':
        """
        Compute the posterior distribution of the gram matrix specified by
        axis after observing data
        """
        return NCWishart(self.mtd.post(self.augment_observed(observed), axis))

    def uni_cond(self, observed) -> tuple:
        """
        One-dimensional conditionals
        """
        all_stats = self.mtd.uni_cond(self.augment_observed(observed))

        return tuple(
            stat['obs', 'obs']
            for stat in all_stats
            )

    def extend(self, observed, gram, axis: int = 0):
        """
        Shorthand for computing posterior and extending along axis
        """
        return self.post(observed, axis).extend(gram, axis)

@dataclass
class NCWishart:
    """
    Wrapper around the block-wishart representation of a conditionned noncentral
    matrix-t
    """
    wishart: Wishart

    def extend(self, new_gram, axis: int = 0) -> NonCentralMatrixT:
        """
        Generate a matrix-t from an existing posterior gram
        """
        return NonCentralMatrixT.from_post(self.wishart, new_gram, axis)
