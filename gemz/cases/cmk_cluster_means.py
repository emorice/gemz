"""
Demonstrate the effects of centering or not the features in CMK models
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from gemz import models
from gemz.cases import case

def write_fig(fd, fig):
    """
    Converts a figure to html and write it to the document
    """
    fig_html = pio.to_html(fig,
        full_html=False
        )
    fd.write(fig_html)

@case
def cmk_cluster_means(output_dir, case_name, report_path):
    """
    Case entry point
    """

    # Data
    # ====

    rng = np.random.default_rng(1234)
    n_samples = 1000

    corr = 0.75
    # N x 2
    linear2d = rng.multivariate_normal(
        np.zeros(2),
        np.array([
            [1., corr],
            [corr, 1.]
            ]),
        size=n_samples
        )

    # Fits
    # ====

    n_clusters = 32
    kmeans_model = models.kmeans.fit(linear2d, n_clusters=n_clusters)

    # Plots
    # =====

    fig_data = go.Figure(
        data = [
            go.Scatter(
                x=linear2d[:, 0],
                y=linear2d[:, 1],
                mode='markers',
                marker={'size': 3},
                name='Data'
                )
            ],
        layout=dict(
            yaxis={'scaleanchor': 'x', 'scaleratio': 1}
            )
        )

    fig_clusters = fig_data.add_trace(
        go.Scatter(
            x=kmeans_model['means'][:, 0],
            y=kmeans_model['means'][:, 1],
            error_x={'array': np.sqrt(kmeans_model['variances'])},
            error_y={'array': np.sqrt(kmeans_model['variances'])},
            mode='markers',
            name='Cluster means'
            )
        )

    with open(report_path, 'w', encoding='utf8') as fd:
        fd.write(case_name)
        write_fig(fd, fig_data)
        write_fig(fd, fig_clusters)
