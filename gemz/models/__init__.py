"""
Module containing all statistical model fit and prediction code
"""
import sys

from . import (
    kmeans, linear, wishart,
    linear_shrinkage,
    linear_shrinkage_cv,
    lscv_precision_target,
    lscv_free_diagonal,
    nonlinear_shrinkage,
    cmk,
    gmm, igmm
    )

from .methods import (
        get,
        fit, predict_loo, eval_loss,
        fit_eval
        )
