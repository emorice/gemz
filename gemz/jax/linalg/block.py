"""
Concrete jax block-array subclasses
"""

from array_api_jax import MetaJaxAPI
from block import BlockMatrix

class JaxBlockMatrix(BlockMatrix):
    """
    Block matrix using jax array as its base type
    """
    aa = MetaJaxAPI
