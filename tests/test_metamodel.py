"""
High-level routines not tied to a specific model
"""

# pylint: disable=unused-import

from test_models import data

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
