"""
Cross-validation utils
"""

import numpy as np

LOSSES = {}

def loss(name):
    """
    Record a named loss function
    """
    def _wrap(function):
        LOSSES[name] = function
        return function
    return _wrap

@loss('RSS')
def rss_loss(method, model, test):
    """
    Evaluate a fitted mode with a classical RSS loss
    """

    predictions = method.predict_loo(model, test)
    return np.sum((test - predictions)**2)

@loss('NAIC')
def naic_loss(method, model, test):
    """
    Negative Akaike Information Criterion loss

    Untested.
    """
    log_pdfs = method.log_pdf(model, test)
    return - np.sum(log_pdfs)

@loss('GEOM')
def geom_loss(method, model, test):
    """
    Geometric aggregate of RSS over dimensions.

    More robust than RSS with heteroskedastic data.
    """
    predictions = method.predict_loo(model, test)
    dim_squares = np.sum((test - predictions)**2, -1)

    return np.sum(np.log(dim_squares))

def fit_cv(data, method, fold_count=10, seed=1234, loss_name=None, **method_kwargs):
    """
    Fit and eval the given method on folds of data

    Args:
        data: N1 x N2. Models are fixed-dimension N1 x N1, and cross-validation
            is performed along N2.
    """

    _, len2 = data.shape
    rng = np.random.default_rng(seed)

    random_rank = rng.choice(len2, len2)

    total_loss = 0.

    loss_name = loss_name or "RSS"
    loss_fn = LOSSES[loss_name]

    for fold_index in range(fold_count):
        in_fold = random_rank % fold_count != fold_index

        fold_model = method.fit(data[..., in_fold], **method_kwargs)

        total_loss += loss_fn(method, fold_model, data[..., ~in_fold])

    return total_loss
