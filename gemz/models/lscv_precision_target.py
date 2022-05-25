"""
Linear shrinkage through cross-validation towards a diagonal precision-adjusted
target
"""

import numpy as np

from gemz.models import linear_shrinkage_cv

def fit(data):
    """
    Fit a diagonally shrunk linear model in two steps.

    First, fit an identity-shrunk model to get a first estimate of precisions.
    Then, build a shrinkage target from it and fit a second shrinkage model
    """

    homoskedastic_cv = linear_shrinkage_cv.fit(data, loss_name="GEOM")

    diag = np.diagonal(homoskedastic_cv['model']['precision'])

    model = linear_shrinkage_cv.fit(data, loss_name='GEOM', target=1. / diag)

    return model

def predict_loo(model, new_data):
    """
    See linear_shrinkage_cv
    """
    return linear_shrinkage_cv.predict_loo(model, new_data)
