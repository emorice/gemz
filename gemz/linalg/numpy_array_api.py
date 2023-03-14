"""
Reference numpy implementation of ArrayAPI
"""

import numpy as np

from .array_api import ArrayAPI

class NumpyArrayAPI(ArrayAPI):
    """
    Reference numpy implementation of ArrayAPI
    """

    @classmethod
    def broadcast_to(cls, array, shape):
        """
        Create new object from existing by broadcasting
        """
        return np.broadcast_to(array, shape)

    @classmethod
    def eye(cls, length):
        """
        Create an identity squaure array of specified dim
        """
        return np.eye(length)
