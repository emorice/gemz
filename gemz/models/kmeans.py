"""
Simple clustering-based predictions.
"""

import numpy as np
import sklearn.cluster

from . import methods
from .cv import Int1dCV

@methods.add('kmeans')
class KMeans:
    """
    K-means clustering and predictions
    """
    @staticmethod
    def fit(data, n_groups):
        """
        Compute clusters, cluster means and dispersion on given data

        Args:
            data: N1 x N2, with N2 being the large dimension to split into clusters
        """

        sk_model = sklearn.cluster.KMeans(n_clusters=n_groups)

        sk_fit = sk_model.fit(data.T)

        # N2
        groups = sk_fit.labels_

        # G x N2
        one_hot = groups == np.arange(n_groups)[:, None]
        # G
        sizes = one_hot.sum(-1)

        # G x N1
        means =  one_hot @ data.T / sizes[:, None]

        centered_data = data - means.T @ one_hot

        variances = one_hot @ (centered_data**2).sum(0) / sizes

        return {
            'n_groups': n_groups,
            'groups': groups,
            'means': means,
            'variances': variances
            }

    @staticmethod
    def predict_loo(model, new_data):
        """
        Leave-one out prediction from existing clusters on new observations

        Args:
            new_data: N1' x N2
        """
        # N2
        groups = model['groups']
        # G
        group_sizes = np.bincount(groups)
        # regularization for empty groups
        group_sizes = group_sizes + 1e-6
        # G x N2
        one_hot = groups == np.arange(model['n_groups'])[:, None]

        # N1' x G, Means of new observations on all samples
        base_means = (
            new_data @ one_hot.T
            / (group_sizes - 1)
            )

        # N1' x N2, LOO correction
        preds = (
            base_means[:, groups]
            - new_data / (group_sizes[groups] - 1)
            )

        return preds

    @staticmethod
    def get_name(spec):
        """
        Readable short name
        """
        return f'{spec["model"]}/{spec["n_groups"]}'

    cv = Int1dCV('n_groups', 'clusters')
