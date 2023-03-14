"""
Module containing all statistical model fit and prediction code
"""
import sys
import pkgutil


from . import (
    kmeans,
    linear,
    wishart,
    linear_shrinkage,
    lscv_precision_target,
    lscv_free_diagonal,
    nonlinear_shrinkage,
    lscv_loo,
    cmk,
    gmm,
    igmm,
    svd,
    cv,
    peer,
    mt_sym,
    )

from .methods import get, get_name

from .ops import (
        fit,
        predict_loo,
        eval_loss,
        fold,
        aggregate_losses,
        cv_residualize,
        )
