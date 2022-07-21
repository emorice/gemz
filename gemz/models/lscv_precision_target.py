"""
Linear shrinkage through cross-validation towards a diagonal precision-adjusted
target
"""

from gemz import models
from gemz.models import cv
from .linear import DualLinear
from .linear_shrinkage_cv import LSCV

@models.add('lscv_precision_target')
class LSCVPrecisionTarget(LSCV):
    """
    Linear shrinkage where target is derived from per-dimension precision
    estimates
    """

    @classmethod
    def fit(cls, data):
        """
        Fit a diagonally shrunk linear model in two steps.

        First, fit an unshrunk model to get a first estimate of precisions.
        Then, build a shrinkage target from it and fit a second shrinkage model
        """

        indiv_rss = cv.fit_cv(data, DualLinear, loss_name='iRSS')

        return super().fit(
            data,
            loss_name='GEOM',
            target=indiv_rss / data.shape[-1]
            )
