"""
Linear shrinkage through cross-validation towards a diagonal precision-adjusted
target
"""

from gemz.models import cv
from . import methods, linear, linear_shrinkage_cv

methods.add_module('lscv_precision_target', __name__)

def fit(data):
    """
    Fit a diagonally shrunk linear model in two steps.

    First, fit an unshrunk model to get a first estimate of precisions.
    Then, build a shrinkage target from it and fit a second shrinkage model
    """

    indiv_rss = cv.fit_cv(data, linear, loss_name='iRSS')

    return linear_shrinkage_cv.fit(
        data,
        loss_name='GEOM',
        target=indiv_rss / data.shape[-1]
        )

predict_loo = linear_shrinkage_cv.predict_loo
