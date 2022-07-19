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

        model_min = {
            'data': data,
            'groups': groups,
            'responsibilities': (probas / probas.sum(-1, keepdims=True)).T
            }

        return GMM.precompute(model_min)

    @staticmethod
    def precompute(model):
        """
        Generate a model with useful precomputed quantities from minimal spec
        """
        data = model['data']
        responsibilities = model['responsibilities']


        # K, sum over P
        group_sizes = np.sum(responsibilities, -1)

        # K x N, sum over P
        means = responsibilities @ data.T / group_sizes[:, None]
        # K x N x N. O(KNP) intermediate, may be improved
        covariances = (
            (data * responsibilities[:, None, :]) @ data.T
                / group_sizes[:, None, None]
            - means[:, :, None] * means[:, None, :]
            ) + 1e-6 * np.eye(data.shape[0]) # TODO: unhardcode
        # K x N X N
        precisions = np.linalg.inv(covariances)

        return {
            **model,
            'group_sizes': group_sizes,
            'means': means,
            'precisions': precisions
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
        means_kn = model['means']
        data_np = model['data']
        precisions_knn = model['precisions']

        centered_data_knp = data_np - means_kn[..., None]

        group_sizes_kp = linalg.loo_sum(resps_kp, -1)

        new_means_kpm = (
            linalg.loo_matmul(resps_kp, new_data_mp.T)
            / group_sizes_kp[..., None]
            )

        grams_kpnm = (
            linalg.loo_cross_square(data_np, resps_kp, new_data_mp)
            / group_sizes_kp[..., None, None]
        )

        covariances_kpnm = linalg.LowRankUpdate(
            grams_kpnm,
            # Dummy contraction + join on loo dim
            means_kn[:, None, :, None],
            -1,
            # Dummy contraction, 2nd to last
            new_means_kpm[..., None, :]
            )

        trans_data_knp = precisions_knn @ centered_data_knp
        trans_data_kp1n = np.swapaxes(trans_data_knp, -2, -1)[..., None, :]

        preds_kpm = (
            new_means_kpm +
            (trans_data_kp1n @ covariances_kpnm)[..., 0, :]
            )

        preds_pm = np.sum(preds_kpm * resps_kp[..., None], 0)

        return preds_pm.T
