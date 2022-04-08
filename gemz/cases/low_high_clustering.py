"""
Demonstrating the difference between clustering in high and low dimensional
settings
"""

import numpy as np
import plotly.graph_objects as go

import gemz.models
from gemz.cases import case
from gemz.reporting import write_fig

def plot_pc_clusters(data, n_clusters):
    """
    Generate a plot visualizing the kmeans clusters on top of the first
    components
    """

    clustering = gemz.models.kmeans.fit(data, n_clusters=n_clusters)
    dev_1d = np.sqrt(clustering['variances'])

    _, _, left_t = np.linalg.svd(data)

    pc1, pc2 = left_t[:2]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=data @ pc1,
                y=data @ pc2,
                mode='markers',
                name='Data',
                marker={'size': 3}
                ),
            go.Scatter(
                x=clustering['means'] @ pc1,
                y=clustering['means'] @ pc2,
                error_x={'array': dev_1d},
                error_y={'array': dev_1d},
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
def low_high_clustering(output_dir, case_name, report_path):
    """
    Case entry point, see module docstring
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

    n_clusters = 12
    fig_low = plot_pc_clusters(low, n_clusters)
    fig_low.update_layout(title='Most information in a few dimensions')

    fig_high = plot_pc_clusters(high, n_clusters)
    fig_high.update_layout(title='Information spread across dimsensions')

    with open(report_path, 'w', encoding='utf8') as fd:
        fd.write(case_name)
        write_fig(fd, fig_low)
        write_fig(fd, fig_high)
