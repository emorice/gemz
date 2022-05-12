"""
Demonstrate the effects of regularising high-dimensional linear models
"""

import numpy as np

import plotly.graph_objects as go

from gemz import models
from gemz.cases import case
from gemz.reporting import write_fig

from gemz.cases.low_high_clustering import plot_pc_clusters

def plot_cv_rss(cv_model, grid_name):
    """
    Generate a scatter plot of the validated rss along the cv search grid
    """
    return go.Figure(
        data=go.Scatter(
            x=cv_model['cv_grid'],
            y=cv_model['cv_rss'],
            mode='lines+markers',
            ),
        layout={
            'title': 'Cross-validated RSS',
            'xaxis': { 'title': grid_name, 'type': 'log'},
            'yaxis_title': 'RSS'
            }
        )

@case
def linear_reg(_, case_name, report_path):
    """
    Case entry point
    """

    # Data
    # ====

    rng = np.random.default_rng(1234)

    # samples = dimension being split by clustering
    # Collected interesting cases:
    # 1000, 297
    # 50, 50
    # 201, 50
    # 201, 200 is really weird ??
    # 201, 196
    n_samples = 201
    n_features = 100

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

        # Superseded by the cv version
        # 'linear_shrinkage': {'prior_var': 100},

        'linear_shrinkage_cv': {
            'prior_var_grid': 10**np.linspace(-2, 2, 20)
            },
        'kmeans': dict(n_clusters=4),
        'nonlinear_shrinkage': {},

        # In progress
        # 'wishart': {}
        }

    model_fits = {
        k: getattr(models, k).fit(train, **kwargs)
        for k, kwargs in model_args.items()
        }

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

    # Plot against first PC
    covariate = np.linalg.svd(train, full_matrices=False)[0][:, 0]

    order = np.argsort(covariate)

    fig_test = go.Figure(
        data=[
            go.Scatter(
                x=covariate[order],
                y=test[:, test_idx][order],
                mode='markers',
                name='New feature'
                )
        ] + [
            go.Scatter(
                x=covariate[order],
                y=pred[order].flatten(),
                mode='lines',
                name=f'{k.capitalize()} prediction'
                )
            for k, pred in preds.items()
            ],
        layout={
            'title': 'Predictions of a new dimension',
            'xaxis': {'title': 'PC1'},
            'yaxis': {'title': 'New dimension'}
            }
        )

    fig_pcs = plot_pc_clusters(
        train,
        n_clusters=model_args['kmeans']['n_clusters']
        )


    spectrum = models.linear.spectrum(train)

    adj_spectrum = models.linear_shrinkage.spectrum(
        train,
        model_fits['linear_shrinkage_cv']['cv_best']
        )

    opt_spectrum = model_fits['nonlinear_shrinkage']['spectrum']

    log1p = False

    fig_spectrum = go.Figure(
        data=[
            go.Scatter(
                y=spec / spec.sum() * spectrum.sum() + 1. * log1p,
                mode='lines+markers',
                name=name,
                )
            for spec, name in [
                (spectrum, 'Covariance spectrum'),
                (adj_spectrum, 'Linearly regularized covariance spectrum'),
                (opt_spectrum, 'Non-Linearly regularized covariance spectrum')
                ]
            ],
        layout={
            'title': 'Spectra',
            'yaxis': {
                'title': 'Eigenvariances' + log1p * ' (log1p)',
                'type': 'log' if log1p else 'linear'}
            }
        )

    with open(report_path, 'w', encoding='utf8') as stream:
        stream.write(case_name)
        write_fig(stream, fig_pcs, fig_test, fig_spectrum,
            plot_cv_rss(model_fits['linear_shrinkage_cv'], 'Prior variance')
            )
