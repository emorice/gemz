"""
Linear shrinkage through cross-validation towards a diagonal precision-adjusted
target
"""

from gemz.models import linear, linear_shrinkage_cv, cv

def fit(data):
    """
    Fit a diagonally shrunk linear model in two steps.

    First, fit an unshrunk model to get a first estimate of precisions.
    Then, build a shrinkage target from it and fit a second shrinkage model
    """

    indiv_rss = cv.fit_cv(data, linear, loss_name='iRSS')

    model = linear_shrinkage_cv.fit(
        data,
        loss_name='GEOM',
        target=indiv_rss / data.shape[-1]
        )

    return model

def predict_loo(model, new_data):
    """
    See linear_shrinkage_cv
    """
    return linear_shrinkage_cv.predict_loo(model, new_data)
