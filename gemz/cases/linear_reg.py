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
                y=pred[order].flatten(),
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

    # N x K, K, K x D
    _, singulars, left_t = np.linalg.svd(train)
    spectrum = singulars**2 / n_samples

    # This shoud just yield spectrum again
    spectrum_mean = np.mean((left_t @ train.T)**2, -1)
    # This is new
    spectrum_var = np.mean((
        (left_t @ train.T)**2
        - spectrum_mean[:, None]
        )**2, -1) / n_samples

    #prior_var = np.exp(model_fits['wishart']['opt']['prior_var_ln'])
    #prior_edf = np.exp(model_fits['wishart']['opt']['prior_edf_ln'])

    #adj_spectrum = (n_samples * spectrum + prior_var) / (n_samples + prior_edf - 1)
    adj_spectrum = spectrum + np.sum(spectrum)/n_samples

    n_sd = 1.
    fig_spectrum = go.Figure(
        data=[
            go.Scatter(
                y=spectrum,
                mode='lines',
                name='Covariance spectrum'
                ),
            go.Scatter(
                #x=np.repeat(np.arange(len(spectrum)), n_samples),
                #y=(left_t @ train.T)**2).flatten(),
                y=spectrum_mean,
                error_y={'array': n_sd*np.sqrt(spectrum_var)},
                #mode='markers',
                name='Re-estimated spectrum'
                ),
            go.Scatter(
                y=spectrum_mean,
                error_y={'array': n_sd*spectrum_mean*np.sqrt(2/n_samples)},
                #mode='markers',
                name='Asymptotic spectrum distribution'
                ),
            go.Scatter(
                y=spectrum_mean,
                error_y={'array':
                n_sd*spectrum_mean*np.sqrt(spectrum_mean.sum()/spectrum_mean*1/n_samples)},
                #mode='markers',
                name='Asymptotic Wishart distribution'
                ),
            go.Scatter(
                y=adj_spectrum,
                mode='lines',
                name='Regularized covariance spectrum'
                )
            ]
        )

    with open(report_path, 'w', encoding='utf8') as stream:
        stream.write(case_name)
        write_fig(stream, fig_pcs, fig_test, fig_spectrum,
            plot_cv_rss(model_fits['linear_shrinkage_cv'], 'Prior variance')
            )
