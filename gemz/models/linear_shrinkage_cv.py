"""
Cross-validation wrappers over linearly shrunk linear model
"""

import numpy as np

from . import linear_shrinkage
from .cv import fit_cv

def default_grid():
    """
    A standard grid of prior vars to screen
    """

    return 10**np.linspace(-2, 2, 20)

def fit(data, prior_var_grid=None, loss_name=None):
    """
    Cross validated linearly regularized precision matrix.

    Basic grid-search strategy.
    """

    if prior_var_grid is None:
        prior_var_grid = default_grid()

    rss_grid = []
    for prior_var in prior_var_grid:
        rss_grid.append(
            fit_cv(data, linear_shrinkage, prior_var=prior_var, loss_name=loss_name)
            )

    best_prior_var = prior_var_grid[
        np.argmin(rss_grid)
        ]

    model = linear_shrinkage.fit(data, prior_var=best_prior_var)

    return {
        'model': model,
        'cv_grid': prior_var_grid,
        'cv_best': best_prior_var,
        'cv_rss': rss_grid
       }

def predict_loo(model, new_data):
    """
    Linear shrinkage loo prediction for the best model found during cv.
    """
    return linear_shrinkage.predict_loo(model['model'], new_data)
