"""
Linear shrinkage through cross-validation towards a diagonal precision-adjusted
target
"""

from . import methods, cv, ops

methods.add_module('lscv_precision_target', __name__)

def fit(data):
    """
    Fit a diagonally shrunk linear model in two steps.

    First, fit an unshrunk model to get a first estimate of precisions.
    Then, build a shrinkage target from it and fit a second shrinkage model
    """

    cfe = ops.cv_fit_eval({'model': 'linear'}, data, loss_name='iRSS')

    return ops.fit(
        {
            'model': 'cv',
            'inner': {
                'model': 'linear_shrinkage',
                'target': cfe['loss'] / data.shape[-1]
                },
            'loss_name': 'GEOM',
            },
        data,
        )

predict_loo = cv.predict_loo
