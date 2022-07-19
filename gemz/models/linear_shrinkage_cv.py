"""
Cross-validation wrappers over linearly shrunk linear model
"""

import numpy as np

from gemz import models
from .cv import fit_cv
from .linear_shrinkage import LinearShrinkage

def default_grid():
    """
    A standard grid of prior vars to screen
    """

    return 10**np.linspace(-2, 2, 20)

@models.add('linear_shrinkage_cv')
class LSCV(LinearShrinkage):
    """
    Linear shrinkage optimized through CV
    """

    @classmethod
    def fit(cls, data, prior_var_grid=None, loss_name=None, target=None):
        """
        Cross validated linearly regularized precision matrix.

        Basic grid-search strategy.
        """

        if prior_var_grid is None:
            prior_var_grid = default_grid()

        loss_grid = []
        for prior_var in prior_var_grid:
            loss_grid.append(
                fit_cv(data, LinearShrinkage, prior_var=prior_var,
                loss_name=loss_name, target=target)
                )

        best_prior_var = prior_var_grid[
            np.argmin(loss_grid)
            ]

        model = super().fit(data, prior_var=best_prior_var, target=target)

        return {
            'model': model,
            'cv_grid': prior_var_grid,
            'cv_best': best_prior_var,
            'cv_loss': loss_grid
           }

    @classmethod
    def predict_loo(cls, model, new_data):
        """
        Linear shrinkage loo prediction for the best model found during cv.
        """
        return super().predict_loo(model['model'], new_data)
