"""
Gaussian mixtrue based predictions.
"""

import numpy as np
import sklearn.mixture

def fit(data, n_groups, bayesian=False, **skargs):
    """
    Compute clusters, cluster means and dispersion on given data
    """
    # data: len1 x len2, len2 is the one to split

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

    sk_fit = sk_model.fit(data.T)

    print('GMM converged: ', sk_fit.converged_)

    # len2
    groups = sk_fit.predict(data.T)
    probas = sk_fit.predict_proba(data.T)

    return {
        'groups': groups,
        'responsibilities': (probas / probas.sum(-1, keepdims=True)).T

        }

def predict_loo(model, new_data):
    """
    Leave-one out prediction from existing mixture components on new observations
    """
    raise NotImplementedError
