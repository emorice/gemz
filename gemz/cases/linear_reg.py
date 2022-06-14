"""
Demonstrate the effects of regularising high-dimensional linear models
"""

import numpy as np

import plotly.graph_objects as go

from gemz import models
from gemz.cases import case
from gemz.reporting import write_fig

from gemz.cases.low_high_clustering import plot_pc_clusters

def gen_hyperv(len1, len2, noise_sd=1., seed=0):
    """
    Generates a high-dimensional "V" shape
    """

    rng = np.random.default_rng(seed)

    # The coordinate along the length of the V
    hidden_factor = rng.normal(0., 1., size=len1)

    # Whether we are in the first or second arm of the V
    hidden_class = hidden_factor > 0.

    # The ends and middle of the V
    support_points = rng.normal(0., 1., size=(3, 2, len2))

    # 2 x  len1 x len2
    signal = support_points[0][:, None, :] + np.where(
        hidden_class[None, :, None],
        + hidden_factor[None, :, None] * support_points[1][:, None, :],
        - hidden_factor[None, :, None] * support_points[2][:, None, :]
        )

    # 2 x len1 x len2
    data = (
        signal
        + rng.normal(0., 1., size=(2, len1, len2))
            * np.array(noise_sd)[..., None]
        )

    # Center each feature over all samples
    data -= np.mean(data, 1)[:, None, :]

    return data

def plot_cv_rss(cv_models, grid_name):
    """
    Generate a scatter plot of the validated rss along the cv search grid
    """
    return go.Figure(
        data=[
            go.Scatter(
                x=cv_model['cv_grid'],
                y=cv_model['cv_rss'],
                yaxis=f'y{i+1 if i else ""}',
                mode='lines+markers',
                )
            for i, cv_model in enumerate(cv_models)
            ],
        layout={
            'title': 'Cross-validated RSS',
            'xaxis': { 'title': grid_name, 'type': 'log'},
            'yaxis_visible': False,
            **{
                f'yaxis{i+1}': {
                    'overlaying': 'y',
                    'visible': False
                    }
                for i in range(1, len(cv_models))
                }
            }
        )

def plot_convergence(fits):
    """
    Plot log_likelihood against iteration, if sensible, for each model
    """
    data = []

    for name, model in fits:
        if 'hist' in model:
            hist = model['hist']
            if 'iteration' in hist and 'log_likelihood' in hist:
                data.append(go.Scatter(
                    x=hist['iteration'],
                    y=hist['log_likelihood'],
                    name=name
                    ))
    return go.Figure(data=data)

@case
def linear_reg(_, case_name, report_path):
    """
    Regularized and unregularized high-dimensional linear models
    """

    # Data
    # ====


    # len1 = dimension being split by clustering
    # Collected interesting cases:
    # 1000, 297
    # 50, 50
    # 201, 50
    # 201, 200 is really weird ??
    # 201, 196

    train, test = gen_hyperv(201, 100, noise_sd=.5)

    # Fits
    # ====

    model_args = [
        ('linear',  {}),

        # Superseded by the cv version
        # ('linear_shrinkage': {'prior_var',  100}),

        ('linear_shrinkage_cv',  {}),
        ('linear_shrinkage_cv', {'loss_name': 'GEOM'}),
        ('kmeans',  {'n_clusters': 4}),
        ('nonlinear_shrinkage',  {}),
        ('wishart',  {}),
        ('cmk', {'n_groups': 1}),
        ('cmk', {'n_groups': 20}),
        ]

    model_fits = [
        (k, getattr(models, k).fit(train, **kwargs))
        for k, kwargs in model_args
        ]

    test_idx = 2
    target = test[:, test_idx]

    # For a new feature, we can make a basic prediction by predicting the mean
    # of all (other) samples of the group

    preds = [
        (k, getattr(models, k).predict_loo(fit, target))
        for k, fit in model_fits
        ]

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
            for k, pred in preds
            ],
        layout={
            'title': 'Predictions of a new dimension',
            'xaxis': {'title': 'PC1'},
            'yaxis': {'title': 'New dimension'}
            }
        )

    fig_pcs = plot_pc_clusters(
        train,
        n_clusters=next(args['n_clusters'] for name, args in model_args
            if name == 'kmeans')
        )


    spectrum = models.linear.spectrum(train)

    adj_spectrum = models.linear_shrinkage.spectrum(
        train,
        next(model['cv_best'] for name, model in model_fits
            if name == 'linear_shrinkage_cv')
        )

    opt_spectrum = next(model['spectrum'] for name, model in model_fits
        if name == 'nonlinear_shrinkage')

    wh_fit = models.wishart.fit(train)
    print(wh_fit)
    wh_spectrum = (
        spectrum
        + np.exp(wh_fit['opt']['prior_var_ln']) / train.shape[-1]
        )

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
                (opt_spectrum, 'Non-Linearly regularized covariance spectrum'),
                (wh_spectrum, 'Lin. reg. covariance spectrum (Wishart EM)')
                ]
            ],
        layout={
            'title': 'Spectra',
            'yaxis': {
                'title': 'Eigenvariances' + log1p * ' (log1p)',
                'type': 'log' if log1p else 'linear'}
            }
        )

    fig_cv = plot_cv_rss([
                model for name, model in model_fits
                if name == 'linear_shrinkage_cv'
            ], 'Prior variance')

    with open(report_path, 'w', encoding='utf8') as stream:
        stream.write(case_name)
        write_fig(stream,
            fig_pcs, fig_test, fig_spectrum,
            fig_cv, plot_convergence(model_fits)
            )
