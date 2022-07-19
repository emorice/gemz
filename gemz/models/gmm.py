"""
Gaussian mixtrue based predictions.
"""

import numpy as np
import sklearn.mixture

from gemz import models
from gemz import linalg

@models.add('gmm')
class GMM:
    """
    Gaussian mixture model, classical EM and variants thereof
    """
    @staticmethod
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
            **skargs,
            )

        sk_fit = sk_model.fit(data.T)

        # len2
        groups = sk_fit.predict(data.T)
        probas = sk_fit.predict_proba(data.T)

        return {
            'groups': groups,
            'responsibilities': (probas / probas.sum(-1, keepdims=True)).T
            }

    @staticmethod
    def predict_loo(model, new_data):
        """
        Args:
            new_data: len1' x len2, with len2 matching training data
        """
        # Index names:
        #  n: len1, small dim
        #  p: len2, large dim
        #  m: len1', new small dim
        #  k: groups

        new_data_mp = new_data

        resps_kp = model['responsibilities']

        group_sizes_kp = linalg.loo_sum(resps_kp, -1)

        new_means_kpm = (
            linalg.loo_matmul(resps_kp, new_data_mp.T)
            / group_sizes_kp[..., None]
            )

        preds_kpm = new_means_kpm

        preds_pm = np.sum(preds_kpm * resps_kp[..., None], 0)

        return preds_pm.T
