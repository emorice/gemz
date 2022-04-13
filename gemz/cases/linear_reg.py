"""
Demonstrate the effects of regularising high-dimensional linear models
"""

import numpy as np

import plotly.graph_objects as go

from gemz import models
from gemz.cases import case
from gemz.reporting import write_fig

from gemz.cases.low_high_clustering import plot_pc_clusters

@case
def linear_reg(output_dir, case_name, report_path):
    """
    Case entry point
    """

    # Data
    # ====

    rng = np.random.default_rng(1234)

    # samples = dimension being split by clustering
    n_samples = 1000
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

    model_args = {
        'linear': {},
        'kmeans': dict(n_clusters=4),
        'wishart': {}
        }

    model_fits = {
        k: getattr(models, k).fit(train, **kwargs)
        for k, kwargs in model_args.items()
        }

    print(model_fits['wishart'])

    test_idx = 2
    target = test[:, test_idx]

    # For a new feature, we can make a basic prediction by predicting the mean
    # of all (other) samples of the group

    preds = {
        k: getattr(models, k).predict_loo(fit, target)
        for k, fit in model_fits.items()
        }

    # Plots
    # =====

    order = np.argsort(hidden_factor)

    fig_test = go.Figure(
        data=[
            go.Scatter(
                x=hidden_factor[order],
                y=test[:, test_idx][order],
                mode='markers',
                name='New feature'
                )
        ] + [
            go.Scatter(
                x=hidden_factor[order],
                y=pred[order],
                mode='lines',
                name=f'{k.capitalize()} prediction'
                )
            for k, pred in preds.items()
            ]
        )

    fig_pcs = plot_pc_clusters(
        train,
        n_clusters=model_args['kmeans']['n_clusters']
        )

    spectrum = (np.linalg.svd(train)[1]**2) / n_samples
    prior_var = np.exp(model_fits['wishart']['opt']['prior_var_ln'])
    prior_edf = np.exp(model_fits['wishart']['opt']['prior_edf_ln'])
    adj_spectrum = (n_samples * spectrum + prior_var) / (n_samples + prior_edf - 1)

    fig_spectrum = go.Figure(
        data=[
            go.Scatter(
                y=spectrum,
                mode='lines',
                name='Covariance spectrum'
                ),
            go.Scatter(
                y=adj_spectrum,
                mode='lines',
                name='Regularized covariance spectrum'
                )
            ]
        )

    with open(report_path, 'w', encoding='utf8') as fd:
        fd.write(case_name)
        write_fig(fd, fig_pcs)
        write_fig(fd, fig_spectrum)
        write_fig(fd, fig_test)
