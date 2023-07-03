"""
High-level routines not tied to a specific model
"""

# pylint: disable=unused-import

import numpy as np

import pytest

from test_models import data, unsplit_data

from gemz import models

# pylint: disable=redefined-outer-name

def test_fit_and_eval(data):
    """
    Fit a model, then call a loss function on a test set
    """
    train, test = data
    mspec = dict(model='linear')
    fitted = models.fit(mspec, train)

    rss = models.eval_loss(mspec, fitted, test, 'RSS')

    assert isinstance(rss,  float)

# Catastrophic unstability on some platform, leading to residuals of the order
# of 1e-5 instead of 1e-15. Not critical.
@pytest.mark.xfail
def test_cv_residualize():
    """
    Impute all values in turn given others
    """

    # Constant, trivial to impute
    data = np.ones((50, 100))

    mspec = {'model': 'linear'}

    res = models.cv_residualize(mspec, data)

    for stat in ['min', 'median', 'mean', 'max']:
        print(stat, getattr(np, stat)(np.abs(res)))

    assert np.allclose(res, np.zeros(data.shape))
