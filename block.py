"""
Block matrices utils
"""

import operator

from dataclasses import dataclass
from typing import Any, TypeGuard
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
    def asarray(cls, obj):
        """
        Convert obj to an api array if necessary
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def broadcast_to(cls, obj, shape):
        """
        Create new object from existing by broadcasting
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def zeros(cls, shape):
        """
        Create an array of specified shape full of zeros
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def eye(cls, length):
        """
        Create an identity squaure array of specified dim
        """
        raise NotImplementedError('Abstract method')

    @classmethod
    def inv(cls, array):
        """
        Applies np.linalg.inv
        """
        return cls.array_function(np.linalg.inv, array)

    @classmethod
    def slogdet(cls, array):
        """
        Applies np.linalg.inv
        """
        return cls.array_function(np.linalg.slogdet, array)

    @classmethod
    def solve(cls, array, rhs):
        """
        Applies np.linalg.solve
        """
        return cls.array_function(np.linalg.solve, array, rhs)

class JaxAPI(ArrayAPI):
    """
    Conversion layer from numpy to jax operations
    """
    functions = {
        np.diagonal: jnp.diagonal,
        np.outer: jnp.outer,
        np.linalg.inv: jnp.linalg.inv,
        np.linalg.slogdet: jnp.linalg.slogdet,
        np.linalg.solve: jnp.linalg.solve,
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
    def asarray(cls, obj):
        return jnp.asarray(obj)

    @classmethod
    def broadcast_to(cls, obj, shape):
        return jnp.broadcast_to(obj, shape)

    @classmethod
    def zeros(cls, shape):
        return jnp.zeros(shape)

    @classmethod
    def eye(cls, length):
        return jnp.eye(length)

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
        if (len(self.dims), len(other.dims)) != (2, 2):
            raise NotImplementedError
        return self.__class__(
                (self.dims[-2], other.dims[-1]),
                dok_product(self.blocks, other.blocks)
                )

    # Numpy protocol
    # --------------

    def __array_function__(self, func, types, args, kwargs):
        del types
        # Functions implemented as methods on first argument
        methods = {
            np.linalg.inv: self._inv,
            np.linalg.slogdet: self._slogdet,
            np.linalg.solve: self.solve,
            np.diagonal: self.diagonal,
            np.swapaxes: self.swapaxes,
            np.transpose: self.transpose,
            np.outer: self._outer,
            np.ndim: lambda: self.ndim,
            }

        if func in methods:
            obj, *other_args = args
            if obj is self:
                return methods[func](*other_args, **kwargs)

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

    def _lu(self):
        """
        Simple block LU.

        This returns two block matrices, L and the strict upper part of U (the
        diagonal being identity matrices)

        Bugs: explicitly inverting the pivots proves to be unstable, as can be
        expected. For now we call solve for each block to eliminate, which will
        factor the pivot several time.
        """
        self._ensure_square()
        if len(self.dims) > 2:
            raise NotImplementedError('Batched block lu')
        lower = dict(self.blocks)
        upper = dict({})

        ordered_kis = tuple(self.dims[-1].keys())

        # Iterate along the diagonal
        for i, kei in enumerate(ordered_kis):
            if (kei, kei) not in lower:
                raise RuntimeError('Block LU with missing diagonal block')
            # Invert pivot
            pivot = lower[kei, kei]
            # Skip for numerical stability tests
            #inv_pivot = self.aa.inv(pivot)

            # Take all blocks in rest of row
            for ki_right in ordered_kis[(i+1):]:
                if (kei, ki_right) in lower:
                    off_block = lower[kei, ki_right]
                    # Delete from lower
                    del lower[kei, ki_right]

                    # Add to upper
                    # todo: save factorization
                    trans_block = self.aa.solve(pivot, off_block)
                    upper[kei, ki_right] = trans_block

                    # Update rest of lower
                    for ki_down in ordered_kis[(i+1):]:
                        if (ki_down, kei) in lower:
                            prod = lower[ki_down, kei] @ trans_block
                            if (ki_down, ki_right) in lower:
                                # Don't use -= here as it could be
                                # misinterpreted as an in-place modification
                                lower[ki_down, ki_right] = (
                                    lower[ki_down, ki_right] - prod
                                    )
                            else:
                                lower[ki_down, ki_right] = - prod
        return self.__class__(self.dims, lower), self.__class__(self.dims, upper)

    def _inv_dense(self):
        """
        Fallback inversion using dense matrices
        """
        self._ensure_square()
        return self.from_dense(
                self.aa.array_function(
                    np.linalg.inv, self.to_dense()
                    ),
                self.dims
                )

    def triangular_solve(self, target, lower=True, id_diag=False):
        """
        Block triangular solver

        This does *not* check if self is triangular of the specified type.
        """
        self._ensure_square()

        result = {}
        tblocks = target.blocks

        key_order = tuple(self.dims[-1].keys())
        if not lower:
            key_order = key_order[::-1]

        # Iterate in order over result row
        for i, key in enumerate(key_order):
            # Iterate over rhs (result col)
            for rhk in target.dims[-1]:
                # Get target block
                if (key, rhk) in tblocks:
                    tblock = tblocks[key, rhk]
                else:
                    tblock = None

                # Eliminate previous coordinates (contracting dim)
                for prev_key in key_order[:i]:
                    if (key, prev_key) not in self.blocks:
                        continue
                    if (prev_key, rhk) not in result:
                        continue
                    update = - (
                        self.blocks[key, prev_key]
                        @ result[prev_key, rhk]
                        )
                    if tblock is None:
                        tblock = update
                    else:
                        tblock = tblock + update
                if tblock is None:
                    continue
                # Solve
                if id_diag:
                    # Fast bypass for diagonal id, as generated by _lu
                    result[key, rhk] = tblock
                else:
                    result[key, rhk] = self.aa.solve(self.blocks[key, key], tblock)

        return self.__class__((self.dims[-2], target.dims[-1]), result)

    def solve(self, target):
        """
        Linear solve. Find x such that self @ x = target
        """
        lower, supper = self._lu()
        inter = lower.triangular_solve(target, lower=True)
        result = supper.triangular_solve(inter, lower=False, id_diag=True)
        return result

    def _inv(self):
        """
        Inversion using simple block LU
        """
        self._ensure_square()
        return self.solve(self.eye(self.dims[-1]))

    def _slogdet(self):
        """
        Simple block determinant by block-LU factorization
        """
        self._ensure_square()
        if len(self.dims) > 2:
            raise NotImplementedError('Batched block lu')

        lower, _supper = self._lu()
        sign = 1.
        logdet = 0.
        for kitem in self.dims[-1]:
            b_sign, b_logdet = self.aa.slogdet(lower[kitem, kitem])
            sign *= b_sign
            logdet += b_logdet
        return sign, logdet


    def _slogdet_dense(self):
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
        Transpose, i.e. reverse the order of all dimensions.
        """
        return self.transpose()

    def transpose(self, axes=None):
        """
        Apply specified permutation to axes, defaulting to reversal
        """
        if axes is None:
            axes = range(self.ndim)[::-1]

        return self.__class__(
                tuple(self.dims[a] for a in axes),
                {
                    tuple(key[a] for a in axes): np.transpose(value, axes)
                    for key, value in self.blocks.items()
                    }
                )

    def swapaxes(self, axis1, axis2):
        """
        Swap the specified axes only.
        """
        perm = list(range(self.ndim))
        perm[axis1] = axis2
        perm[axis2] = axis1
        return self.transpose(perm)

    @property
    def ndim(self):
        """
        Number of block dimensions.

        Actual blocks may be larger dimensional
        """
        return len(self.dims)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Total shape, summing over blocks shapes
        """
        return dims_to_shape(self.dims)

    # Creation and conversion methods
    # -------------------------------

    @classmethod
    def asarray(cls, obj):
        """
        Extends the array api to pass self types unmodified

        The behavoir when various subclasses of BlockMatrix are involved is yet
        to be specified
        """
        if isinstance(obj, BlockMatrix):
            return obj
        return cls.aa.asarray(obj)

    @classmethod
    def indexes(cls, array):
        """
        Unified interface to generalized shape
        """
        if isinstance(array, BlockMatrix):
            return array.dims
        return np.shape(array)

    @classmethod
    def broadcast_to(cls, array, dims: Dims):
        """
        Extension of api's broadcast to to handle generalized dims
        """
        if is_named(dims):
            raise NotImplementedError
        return cls.aa.broadcast_to(array, dims)

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
    def from_dense(cls, dense, dims: Dims | None = None):
        """
        Split array in blocks conforming to dims

        If dims is not provided, make each scalar element a block. Mostly used
        for testing purposes.

        If dims is provided and is not a named shape, pass dense unmodifed,
        without performing any shape checking.
        """
        dense = cls.asarray(dense)
        if dims is None:
            dims = tuple(
                {i: 1 for i in range(length)}
                for length in np.shape(dense)
                )

        if is_named(dims):
            cumshape = dims_to_cumshape(dims)
            blocks: dict = {}

            for key in itertools.product(*dims):
                indexer = _key_to_indexer(key, dims, cumshape)
                lower_dims = tuple(dim[ki] for dim, ki in zip(dims, key))
                blocks[key] = cls.from_dense(dense[indexer], lower_dims)

            return cls(dims, blocks)
        return dense

    @classmethod
    def from_blocks(cls, blocks: dict):
        """
        Build matrix from dictionnary of blocks, inferring dimensions

        Number of dimensions is the max of any key or block, all the rest are
        broadcasted by adding extra dimensions on the left.

        In constrast to blocks, all keys must however have the same length,
        implictly setting several blocks by specifying partial keys is not
        supported. This may change in the future.
        """
        # First pass: compute number of dims
        key_len = max(map(len, blocks.keys()))
        if not all(len(key) == key_len for key in blocks.keys()):
            raise TypeError('All keys must have the same length')
        ndims = max(np.ndim(val) for val in blocks.values())
        ndims = max(ndims, key_len)
        dims : tuple[dict[Any, Dim], ...] = tuple({} for i in range(ndims))

        # Harmonize key dimensions but don't touch blocks yet
        blocks = { _expand_tuple(key, ndims, 0): val
                  for key, val in blocks.items() }

        # Second pass: compute block shapes
        for key, val in blocks.items():
            block_dims = _expand_tuple(cls.indexes(val), ndims, 1)
            for dim, key_item, block_dim in zip(dims, key, block_dims):
                if key_item not in dim:
                    dim[key_item] = block_dim
                else:
                    dim[key_item] = broadcast_dim(dim[key_item], block_dim)

        # Final pass: broadcast each block to its inferred shape
        bcast_blocks = {}
        for key, val in blocks.items():
            block_dims = tuple(dims[i][k] for i, k in enumerate(key))
            bcast_blocks[key] = np.broadcast_to(val, block_dims)

        return cls(dims, bcast_blocks)

    @classmethod
    def eye(cls, dim: Dim):
        """
        Identity matrix of given generalized dim.
        """
        if isinstance(dim, Mapping):
            return cls(
                (dim, dim),
                { (ki, ki): cls.eye(sub_dim)
                 for ki, sub_dim in dim.items() }
                 )
        return cls.aa.eye(dim)

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
    dictionnaries and not a tuple of ints. An empty tuple is not considered a
    NamedDims.
    """
    return len(dims) > 0 and all(isinstance(dim, Mapping) for dim in dims)

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

def broadcast_dim(dim1: Dim, dim2: Dim) -> Dim:
    """
    Generalized broadcasting for dim elements (ints and dicts).

    Regular broadcasting means length 1 and n can be mapped to n and n.

    Here, on top, broadcasting 1 against a named dim is allowed, and
    broadcasting two named dims recursively broadcasts children.
    """
    if isinstance(dim1, int):
        if isinstance(dim2, int):
            # Regular integer broadcast
            if dim1 == 1 or dim2 == 1 or dim1 == dim2:
                return max(dim1, dim2)
        else:
            # Int to named
            if dim1 == 1:
                return dim2
    else:
        if isinstance(dim2, int):
            # Named to int
            if dim2 == 1:
                return dim1
        else:
            # Named to named
            if dim1.keys() == dim2.keys():
                return { k: broadcast_dim(val, dim2[k])
                        for k, val in dim1.items() }

    raise TypeError(f'Incompatible dims: {dim1} and {dim2}')

def _expand_tuple(tup: tuple, length: int, filler) -> tuple:
    return tuple((
        *(filler for _ in range(length - len(tup))),
        *tup
        ))

def broadcast_dims(dims1: Dims, dims2: Dims):
    """
    Broadcast generalized shapes
    """
    ndims = max(len(dims1), len(dims2))
    dims1 = _expand_tuple(dims1, ndims, 1)
    dims2 = _expand_tuple(dims2, ndims, 1)
    return tuple(broadcast_dim(dim1, dim2) for dim1, dim2 in zip(dims1, dims2))

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
