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
    data = signal + rng.normal(0., .5, size=(2, n_samples, n_features))

    # Center each feature over all samples
    data -= np.mean(data, 1)[:, None, :]

    train, test = data

    # Fits
    # ====

    n_clusters = 4

    kmeans_model = models.kmeans.fit(train, n_clusters=n_clusters)

    _, _, left_t = np.linalg.svd(train)
    pc_nl = left_t[0]

    test_idx = 1
    target = test[:, test_idx]

    # For a new feature, we can make a basic prediction by predicting the mean
    # of all (other) samples of the group

    group_sizes = np.bincount(kmeans_model['groups'])
    test_means = (
        np.bincount(kmeans_model['groups'], weights=target)
        / (group_sizes - 1)
        )
    test_means_preds = (
        test_means[kmeans_model['groups']]
        - target / (group_sizes[kmeans_model['groups']] - 1)
        )

    # We can also use a simple linear model
    ## inverse correlation between features, from all samples
    base_prec = np.linalg.inv(train.T @ train)
    ## corr between features and target, idem
    base_covs = train.T @ target
    ## per-sample LOO corrs
    covs = base_covs[:, None] - train.T * target
    ## still biases by use of self in prec
    base_weights = base_prec @ covs
    weights = (
        base_weights
        + (base_prec @ train.T)
            * np.sum((base_prec @ train.T) * covs, 0)
            / (1. - np.sum(train.T * (base_prec @ train.T), 0))
        )
    test_lin = np.sum(train.T * weights, 0)

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
                y=test_means_preds[order],
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
