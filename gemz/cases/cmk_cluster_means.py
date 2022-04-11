"""
Demonstrate the effects of centering or not the features in CMK models
"""

import numpy as np

#from gemz import models
from gemz.cases import case
from gemz.reporting import write_fig

from gemz.cases.low_high_clustering import plot_pc_clusters

@case
def cmk_cluster_means(output_dir, case_name, report_path):
    """
    Case entry point
    """

    # Data
    # ====

    rng = np.random.default_rng(1234)
    n_samples = 100
    n_features = 30

    hidden_factor = rng.normal(0., 1., size=n_samples)
    hidden_class = hidden_factor > 0.

    #corr = 0.9

    support_points = rng.normal(0., 1., size=(3, 2, n_features))

    # 2 x  n_samples x n_features
    signal = support_points[0][:, None, :] + np.where(
        hidden_class[None, :, None],
        + hidden_factor[None, :, None] * support_points[1][:, None, :],
        - hidden_factor[None, :, None] * support_points[2][:, None, :]
        )

    # 2 x n_samples x n_features
    data = signal + rng.normal(0., 0.2, size=(2, n_samples, n_features))

    train, test = data

    # Fits
    # ====

    n_clusters = 2

    #kmeans_model = models.kmeans.fit(data, n_clusters=n_clusters)

    # Plots
    # =====

    fig_pcs = plot_pc_clusters(train, n_clusters=n_clusters)

    with open(report_path, 'w', encoding='utf8') as fd:
        fd.write(case_name)
        write_fig(fd, fig_pcs)
