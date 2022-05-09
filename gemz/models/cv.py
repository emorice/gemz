"""
Cross-validation utils
"""

import numpy as np

def fit_cv(data, method, fold_count=10, seed=1234, **method_kwargs):
    """
    Fit and eval the given method on folds of data

    Args:
        data: N1 x N2. Models are fixed-dimension N1 x N1, and cross-validation
            is performed along N2.
    """

    _, len2 = data.shape
    rng = np.random.default_rng(seed)

    random_rank = rng.choice(len2, len2)

    rss = 0.

    for fold_index in range(fold_count):
        in_fold = random_rank % fold_count != fold_index

        fold_model = method.fit(data[..., in_fold], **method_kwargs)

        fold_test = data[..., ~in_fold]
        fold_predictions = method.predict_loo(fold_model, fold_test)

        rss += np.sum((fold_test - fold_predictions)**2)

    return rss
