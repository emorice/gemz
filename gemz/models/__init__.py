"""
Module containing all statistical model fit and prediction code
"""
import sys

# TODO: this should have some discovery scheme
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

def get(name):
    """
    Returns a model by name
    """
    return getattr(sys.modules[__name__], name)
