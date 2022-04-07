"""
Simple clustering-based predictions.
"""

import numpy as np
import sklearn.cluster

def fit(data, n_clusters):
    """
    Compute clusters, cluster means and dispersion on given data
    """
    # data: N x D

    sk_model = sklearn.cluster.KMeans(n_clusters=n_clusters)

    sk_fit = sk_model.fit(data)

    # N
    groups = sk_fit.labels_

    # G x N
    one_hot = groups == np.arange(n_clusters)[:, None]
    # G
    sizes = one_hot.sum(-1)

    # G x D
    means =  one_hot @ data / sizes[:, None]

    centered_data = data - one_hot.T @ means

    variances = one_hot @ (centered_data**2).sum(-1) / sizes

    return {
        'groups': groups,
        'means': means,
        'variances': variances
        }
