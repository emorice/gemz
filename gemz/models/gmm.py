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

        # Sklearn applies a default reg if not given and we need it again at
        # prediction time
        reg_covar = skargs['reg_covar'] if 'reg_covar' in skargs else 1e-6

        model_min = {
            'data': data,
            'groups': groups,
            'responsibilities': (probas / probas.sum(-1, keepdims=True)).T,
            'reg_covar': reg_covar,
            }

        return GMM.precompute_loo(model_min)

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
            )
        # K x N X N
        precisions = np.linalg.inv(covariances)

        return {
            **model,
            'group_sizes': group_sizes,
            'means': means,
            'precisions': precisions
            }

    @staticmethod
    def precompute_loo(model):
        """
        Generate a model with useful LOO precomputed quantities from minimal spec
        """
        data_np = model['data']
        resps_kp = model['responsibilities']

        # If no reg was needed at fit time we assume none at predict time
        reg_covar = model['reg_covar'] if 'reg_covar' in model else 0.

        group_sizes_kp = linalg.loo_sum(resps_kp, -1)

        means_kpn = linalg.loo_matmul(resps_kp, data_np.T) / group_sizes_kp[..., None]

        # FIXME: reg_covar is applied to the square not the covar
        grams_kpnn = (
            linalg.loo_square(data_np, resps_kp, reg=reg_covar * data_np.shape[-1])
            / group_sizes_kp[..., None, None]
            )

        covariances_kpnn = linalg.SymmetricLowRankUpdate(
            base=grams_kpnn,
            # Extra contracting dim (outer product)
            factor=means_kpn[..., None],
            weight=-1,
            )

        precisions_kpnn = np.linalg.inv(covariances_kpnn)

        return {
            **model,
            'group_sizes': group_sizes_kp,
            'means': means_kpn,
            'precisions': precisions_kpnn
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

        data_np = model['data']
        resps_kp = model['responsibilities']
        group_sizes_kp = model['group_sizes']
        means_kpn = model['means']
        precisions_kpnn = model['precisions']

        new_means_kpm = (
            linalg.loo_matmul(resps_kp, new_data_mp.T)
            / group_sizes_kp[..., None]
            )

        new_grams_kpmn = (
            linalg.loo_cross_square(new_data_mp, resps_kp, data_np)
            / group_sizes_kp[..., None, None]
        )

        new_covariances_kpmn = linalg.LowRankUpdate(
            new_grams_kpmn,
            # Dummy contraction, last
            new_means_kpm[..., None], # kpm1
            -1,
            # Dummy contraction, 2nd to last
            means_kpn[..., None, :] # kp1n
            )

        centered_data_kpn = data_np.T - means_kpn

        trans_data_kpn1 = precisions_kpnn @ centered_data_kpn[..., None]

        preds_kpm = (
            new_means_kpm +
            (new_covariances_kpmn @ trans_data_kpn1)[..., 0] # kpmn kpn1 -> kpm1
            )

        preds_pm = np.sum(preds_kpm * resps_kp[..., None], 0)

        return preds_pm.T
