"""
Demonstrate the effects of centering or not the features in CMK models
"""

import numpy as np

import plotly.graph_objects as go

from gemz import models
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

    # samples = dimension being split by clustering
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

    kmeans_model = models.kmeans.fit(train, n_clusters=n_clusters)

    _, _, left_t = np.linalg.svd(train)
    pc_nl = left_t[1]

    test_idx = 1
    target = test[:, test_idx]

    # For a new feature, we can make a basic prediction by predicting the mean
    # of all (other) samples of the group

    test_means = (
        np.bincount(kmeans_model['groups'], weights=target)
        / np.bincount(kmeans_model['groups'])
        )

    # We can also use a simple linear model
    test_lin = train @ np.linalg.solve(train.T @ train, train.T @ target)

    # Plots
    # =====

    order = np.argsort(train @ pc_nl)

    fig_test = go.Figure(
        data=[
            go.Scatter(
                x=(train @ pc_nl)[order],
                y=test[:, test_idx][order],
                mode='markers',
                name='New feature'
                ),
            go.Scatter(
                x=(train @ pc_nl)[order],
                y=test_lin[order],
                mode='lines',
                name='Linear prediction'
                ),
            go.Scatter(
                x=(train @ pc_nl)[order],
                y=test_means[kmeans_model['groups']][order],
                mode='lines',
                name='K-means prediction'
                )
            ]
        )

    fig_pcs = plot_pc_clusters(train, n_clusters=n_clusters)

    with open(report_path, 'w', encoding='utf8') as fd:
        fd.write(case_name)
        write_fig(fd, fig_pcs)
        write_fig(fd, fig_test)
