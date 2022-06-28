"""
Demonstrating the difference between clustering in high and low dimensional
settings
"""

import numpy as np
import plotly.graph_objects as go

import gemz.models
from gemz.cases import case
from gemz.reporting import open_report, write_fig

def plot_pc_clusters(data, n_clusters):
    """
    Generate a plot visualizing the kmeans clusters on top of the first
    components
    """

    clustering = gemz.models.kmeans.fit(data, n_clusters=n_clusters)

    _, _, left_t = np.linalg.svd(data, full_matrices=False)

    pcs = left_t[:2]

    stds = []
    for i in range(2):
        centered = data @ pcs[i] - np.take_along_axis(
            clustering['means'] @ pcs[i],
            clustering['groups'],
            0
            )
        l2s = np.bincount(
            clustering['groups'],
            weights=centered**2)
        stds.append(np.sqrt(
            l2s / np.bincount(clustering['groups'])
            ))

    fig = go.Figure(
        data=[
            go.Scatter(
                x=data @ pcs[0],
                y=data @ pcs[1],
                mode='markers',
                name='Data',
                marker={'size': 3}
                ),
            go.Scatter(
                x=clustering['means'] @ pcs[0],
                y=clustering['means'] @ pcs[1],
                error_x={'array': stds[0]},
                error_y={'array': stds[1]},
                mode='markers',
                name='Cluster means',
                )
            ],
        layout=dict(
            yaxis={
                'title': 'PC2',
                'scaleanchor': 'x', 'scaleratio': 1.},
            xaxis={
                'title': 'PC1'
                }
            )
        )

    return fig

@case
def low_high_clustering(_, case_name, report_path):
    """
    Clustering of points from an essentially low or high dimensional
    distribution
    """

    rng = np.random.default_rng(1234)
    n_samples = 1000

    low_d = 2
    low = rng.multivariate_normal(
        np.ones(low_d),
        np.array([
            [10., 0.],
            [0., 1.]
            ]),
            size=n_samples
            )

    high_d = 1000
    high = rng.normal(0., 1., size=(n_samples, high_d))

    n_clusters = 33
    fig_low = plot_pc_clusters(low, n_clusters)
    fig_low.update_layout(title='Most information in a few dimensions')

    fig_high = plot_pc_clusters(high, n_clusters)
    fig_high.update_layout(title='Information spread across dimensions')

    with open_report(report_path, case_name) as stream:
        write_fig(stream, fig_low)
        write_fig(stream, fig_high)
