"""
Demonstrate the effects of regularising high-dimensional linear models
"""

import numpy as np

import plotly.graph_objects as go

from gemz import models
from gemz.cases import case
from gemz.reporting import write_fig
from gemz.plots import plot_cv

from gemz.cases.low_high_clustering import plot_pc_clusters

def gen_hyperv(len1, len2, noise_sd=1., seed=0):
    """
    Generates a high-dimensional "V" shape with len2 dimensions, and len1
    replicates.

    Args:
        noise_sd: len2 noise scales for each dimension
    """

    rng = np.random.default_rng(seed)

    # The coordinate along the length of the V
    hidden_factor = rng.normal(0., 1., size=len2)

    # Whether we are in the first or second arm of the V
    hidden_class = hidden_factor > 0.

    # The ends and middle of the V
    support_points = rng.normal(0., 1., size=(3, 2, len1))

    # 2 x  len1 x len2
    signal = support_points[0][..., None] + np.where(
        hidden_class,
        + hidden_factor * support_points[1][..., None],
        - hidden_factor * support_points[2][..., None]
        )

    # 2 x len1 x len2
    data = (
        signal
        + rng.normal(0., 1., size=(2, len1, len2))
            * np.array(noise_sd)
        )

    # Center each feature over all samples
    data -= np.mean(data, -1, keepdims=True)

    return data

def plot_convergence(spec, fit):
    """
    Plot loss against iteration
    """
    hist = np.array(fit['opt']['hist'])
    return go.Figure(
        data=[
            go.Scatter(
                y=np.minimum.accumulate(hist),
                mode='lines',
                name='Current minimum'
                ),
            go.Scatter(
                y=hist,
                mode='markers',
                name='All values',
                ),
            ],
        layout={
            'title': f'Convergence behavior for {models.get_name(spec)}',
            'xaxis_title': 'Iteration',
            'yaxis_title': 'Loss'
            }
        )

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

    train, test = gen_hyperv(100, 251, noise_sd=.5)

    # Fits
    # ====

    model_args = [
        ('linear',  {}),
        ('mt_sym', {'centered': (True, True)}),
        ('cv', {'inner': {'model': 'linear_shrinkage'}}),
        ('cv', {'inner': {'model': 'linear_shrinkage'}, 'loss_name': 'GEOM'}),
        ('lscv_loo', {}),
        ('lscv_loo', {'loss': 'indep'}),
        ('lscv_loo', {'loss': 'joint'}),
        ('kmeans', {'n_groups': 10}),
        ('nonlinear_shrinkage',  {}),
        ('wishart',  {}),
        ('cmk', {'n_groups': 1}),
        ('cmk', {'n_groups': 20}),
        ('gmm', {'n_groups': 2}),
        ('igmm', {'n_groups': 2}),
        ('svd', {'n_factors': 4}),
        ('peer', {'n_factors': 4}),
        ('peer', {'n_factors': 4, 'reestimate_precision': True}),
        ('cv', {'inner': {'model': 'svd'}}),
        # Slow
        # ('cv', {'inner': {'model': 'peer'}, 'grid': np.arange(1, 20)}),
        # ('cv', {'inner': {'model': 'cmk'}}),
        # ('cv', {'inner': {'model': 'gmm'}}),
        # ('cv', {'inner': {'model': 'igmm'}}),
        # ('cv', {'inner': {'model': 'kmeans'}}),
        ]

    model_specs = [
            {
                'model': name,
                **args
                }
            for name, args in model_args
            ]

    model_fits = [
            (k, models.fit({'model': k, **kwargs}, train))
        for k, kwargs in model_args
        ]

    test_idx = 2
    target = test[test_idx]

    # For a new feature, we can make a basic prediction by predicting the mean
    # of all (other) samples of the group

    preds = [
        (k, models.get(k).predict_loo(fit, target[None, :])[0])
        for k, fit in model_fits
        ]

    # Plots
    # =====

    # Plot against first PC
    covariate = np.linalg.svd(train, full_matrices=False)[-1][0]

    order = np.argsort(covariate)

    fig_test = go.Figure(
        data=[
            go.Scatter(
                x=covariate[order],
                y=test[test_idx][order],
                mode='markers',
                name='New feature'
                )
        ] + [
            go.Scatter(
                x=covariate[order],
                y=pred[order].flatten(),
                mode='lines',
                name=f'{models.get_name(spec)}'
                )
            for (k, pred), spec in zip(preds, model_specs)
            ],
        layout={
            'title': 'Predictions of a new dimension',
            'xaxis': {'title': 'PC1'},
            'yaxis': {'title': 'New dimension'}
            }
        )

    fig_pcs = plot_pc_clusters(
        train.T,
        n_clusters=4,
        )


    spectrum = models.get('linear').spectrum(train)

    adj_spectrum = models.get('linear_shrinkage').spectrum(
        train,
        next(model['selected']['prior_var']
            for name, model in model_fits
            if name == 'cv'
            if model['inner']['model'] == 'linear_shrinkage'
            )
        )

    opt_spectrum = next(model['spectrum'] for name, model in model_fits
        if name == 'nonlinear_shrinkage')

    wh_fit = models.get('wishart').fit(train)
    wh_spectrum = (
        spectrum
        + np.exp(wh_fit['opt']['opt']['prior_var_ln']) / train.shape[-1]
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

    figs_cv = sum((
            plot_cv(spec, model_fit)
                for spec, (name, model_fit) in zip(model_specs, model_fits)
                if name == 'cv'
                ), start=[])

    figs_cg = [
            plot_convergence(spec, model_fit)
                for spec, (name, model_fit) in zip(model_specs, model_fits)
                if 'opt' in model_fit
            ]

    with open(report_path, 'w', encoding='utf8') as stream:
        stream.write(case_name)
        write_fig(stream,
            fig_pcs, fig_test, fig_spectrum,
            *figs_cv, *figs_cg
            )
