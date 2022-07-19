"""
Linear model with a linear shrinkage
"""

import numpy as np

from gemz import linalg, models
from .linear import Linear

@models.add('linear_shrinkage')
class LinearShrinkage(Linear):
    """
    Linear shrinkage
    """
    @staticmethod
    def fit(data, prior_var, target=None, scale=None):
        """
        Computes a representation of a linearily regularized precision matrix

        Args:
            data: N1 x N2, assuming N1 < N2
            scale: N2
        Returns:
            Representation of a N2 x N2 regularized precision matrix
        """

        len1, len2 = data.shape

        if target is None:
            target = 1.

        if scale is None:
            scale = 1.

        scale = scale + np.zeros(len2)

        regularized_covariance = linalg.SymmetricLowRankUpdate(
            prior_var * target,
            (scale * data).T,
            1./len1)

        return {
            'precision': np.linalg.inv(regularized_covariance),
            'prior_var': prior_var
            }

    @classmethod
    def spectrum(cls, data, prior_var):
        """
        Spectrum implicitely used by the regularized model when target and scale are None
        """
        orig_spectrum = super().spectrum(data)
        adjusted_spectrum = orig_spectrum + prior_var
        return adjusted_spectrum
