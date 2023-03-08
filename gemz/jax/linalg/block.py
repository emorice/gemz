"""
Concrete jax block-array subclasses
"""

from .array_api import MetaJaxAPI
from gemz.linalg.block import BlockMatrix

class JaxBlockMatrix(BlockMatrix):
    """
    Block matrix using jax array as its base type
    """
    aa = MetaJaxAPI
