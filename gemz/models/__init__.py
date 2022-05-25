"""
Module containing all statistical model fit and prediction code
"""
import sys

from . import (
    kmeans, linear, wishart,
    linear_shrinkage,
    linear_shrinkage_cv,
    lscv_precision_target,
    nonlinear_shrinkage
    )

def get(name):
    """
    Returns a model by name
    """
    return getattr(sys.modules[__name__], name)
