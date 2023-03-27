"""
Demonstrate the effects of regularising high-dimensional linear models
"""

import numpy as np

import plotly.graph_objects as go

from gemz import models
from gemz.cases import case, Case, Output
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

class LinearReg(Case):
    """
    Regularized and unregularized high-dimensional linear models
    """
    name = 'linear_reg'

    @property
    def model_specs(self):
        return [
            {'model': 'linear',	},
            {'model': 'mt_sym',	'centered': (True, True)},
            {'model': 'cv',	'inner': {'model': 'linear_shrinkage'}},
            {'model': 'cv',	'inner': {'model': 'linear_shrinkage'}, 'loss_name': 'GEOM'},
            {'model': 'lscv_loo',	},
            {'model': 'lscv_loo',	'loss': 'indep'},
            {'model': 'lscv_loo',	'loss': 'joint'},
            {'model': 'kmeans',	'n_groups': 10},
            {'model': 'nonlinear_shrinkage',	},
            {'model': 'wishart',	},
            {'model': 'cmk',	'n_groups': 1},
            {'model': 'cmk',	'n_groups': 20},
            {'model': 'gmm',	'n_groups': 2},
            {'model': 'igmm',	'n_groups': 2},
            {'model': 'svd',	'n_factors': 4},
            {'model': 'peer',	'n_factors': 4},
            {'model': 'peer',	'n_factors': 4, 'reestimate_precision': True},
            {'model': 'cv',	'inner': {'model': 'svd'}},
            ]
            # Slow
            # {'model': 'cv',	'inner': {'model': 'peer'}, 'grid': np.arange(1, 20)},
            # {'model': 'cv',	'inner': {'model': 'cmk'}},
            # {'model': 'cv',	'inner': {'model': 'gmm'}},
            # {'model': 'cv',	'inner': {'model': 'igmm'}},
            # {'model': 'cv',	'inner': {'model': 'kmeans'}},

    def __call__(self, output: Output, model_specs=None):
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

        if model_specs is None:
            model_specs = self.model_specs

        model_fits = [
                (models.get_name(spec), models.fit(spec, train))
                for spec in model_specs
                ]

        test_idx = 2
        target = test[test_idx]

        # For a new feature, we can make a basic prediction by predicting the mean
        # of all (other) samples of the group

        preds = [
            (name, models.predict_loo(spec, fit, target[None, :])[0])
            for spec, (name, fit) in zip(model_specs, model_fits)
            ]

        # Plots
        # =====

        # Plot against first PC
        covariate = np.linalg.svd(train, full_matrices=False)[-1][0]

        order = np.argsort(covariate)

        output.add_figure(
                plot_pc_clusters(
                    train.T,
                    n_clusters=4,
                    )
                )

        output.add_figure(go.Figure(
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
                    name=name,
                    )
                # FIXME: this is the loop we want to factor out
                for (name, pred) in preds
                ],
            layout={
                'title': 'Predictions of a new dimension',
                'xaxis': {'title': 'PC1'},
                'yaxis': {'title': 'New dimension'}
                }
            ))

        spectrum = models.get('linear').spectrum(train)

        adj_spectrum = None
        opt_spectrum = None
        wh_spectrum = None
        for name, model in model_fits:
            if name == 'cv' and model['inner']['model'] == 'linear_shrinkage':
                adj_spectrum = models.get('linear_shrinkage').spectrum(
                    train, model['selected']['prior_var']
                    )
            elif name == 'nonlinear_shrinkage':
                opt_spectrum = model['spectrum']
            elif name == 'wishart':
                wh_spectrum = (
                    spectrum
                    + np.exp(model['opt']['opt']['prior_var_ln']) / train.shape[-1]
                    )

        log1p = False

        output.add_figure(go.Figure(
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
                if spec is not None
                ],
            layout={
                'title': 'Spectra',
                'yaxis': {
                    'title': 'Eigenvariances' + log1p * ' (log1p)',
                    'type': 'log' if log1p else 'linear'}
                }
            ))

        for spec, (name, model_fit) in zip(model_specs, model_fits):
            if spec['model'] != 'cv':
                continue
            for fig in plot_cv(spec, model_fit):
                output.add_figure(fig)

        for spec, (name, model_fit) in zip(model_specs, model_fits):
            if 'opt' not in model_fit:
                continue
            output.add_figure(plot_convergence(spec, model_fit))
