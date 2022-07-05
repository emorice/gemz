"""
Gaussian mixtrue based predictions.
"""

import numpy as np
import sklearn.mixture

def fit(data, n_groups, bayesian=False, **skargs):
    """
    Compute clusters, cluster means and dispersion on given data
    """
    # data: len1 x len2, len1 is the one to split

    if bayesian:
        Mixture = sklearn.mixture.BayesianGaussianMixture
    else:
        Mixture = sklearn.mixture.GaussianMixture
    sk_model = Mixture(
        n_components=n_groups,
        covariance_type='full',
        random_state=1,
        verbose=2,
        verbose_interval=1,
        **skargs,
        )

    sk_fit = sk_model.fit(data)

    print('GMM converged: ', sk_fit.converged_)

    # len1
    groups = sk_fit.predict(data)

    return {
        'groups': groups,
        }

def predict_loo(model, new_data):
    """
    Leave-one out prediction from existing mixture components on new observations
    """
    raise NotImplementedError
