"""
Heterogenous signal-to-noise ratios
"""

import numpy as np

from plotly import graph_objects as go
from plotly.subplots import make_subplots

from gemz.cases import case
from gemz.cases.linear_reg import gen_hyperv
from gemz.reporting import write_header, write_footer, write_fig

from gemz.models import linear_shrinkage_cv, linear_shrinkage

def plot_data(data, noise_class):
    """
    Initial exploratory plot
    """

    pc1, pc2 = np.linalg.svd(data, full_matrices=False)[0][:, :2].T

    return go.Figure(
        data=[
            go.Scatter(
                x=pc1[subset],
                y=pc2[subset],
                mode='markers',
                name=subname,
                )
            for subset, subname in [
               (~noise_class, 'Low noise'),
               (noise_class, 'High noise'),
               ]
            ],
        layout={
            'title': 'First sample components for each variable',
            'xaxis_title': 'PC1',
            'yaxis_title': 'PC2',
            'height': 800
            }
        )

def plot_fits(data, target, subsets, predictions):
    """
    Plot various fits
    """

    covariate = np.linalg.svd(data, full_matrices=False)[0][:, 0]

    order = np.argsort(covariate)
    rev_order = np.argsort(order)

    fig = make_subplots(3, 2)

    for i_fit, (k_fit, fit_predictions) in enumerate(predictions.items()):
        for i_refit, (k_refit, refit_predictions) in enumerate(fit_predictions.items()):
            subset = subsets[k_refit]
            imp_order = np.argsort(rev_order[subset])

            sub_covariate = covariate[subset][imp_order]

            coos = i_fit + 1, i_refit + 1
            fid = i_fit * len(fit_predictions) + i_refit
            fid = fid + 1 if fid else ''

            for sub, color in [
                (~subset, 'lightgrey'),
                (subset, 'black'),
                ]:
                fig.add_trace(
                    go.Scatter(
                        x=covariate[sub],
                        y=target[sub],
                        mode='markers',
                        showlegend=False,
                        marker={'color': color}
                    ),
                    *coos)
            fig.add_trace(
                go.Scatter(
                    x=sub_covariate,
                    y=refit_predictions.flatten()[imp_order],
                    mode='lines',
                    name=f'{k_fit} => {k_refit}',
                    showlegend=False,
                    line={'color': 'red'}
                    ),
                *coos)
            fig.update_layout({
                f'xaxis{fid}_title': 'PC1',
                f'yaxis{fid}_title': 'New observations',
                })
            fig.add_annotation(
                {'text': f'{k_fit} \u2192 {k_refit}'},
                yref=f'y{fid} domain',
                xref=f'x{fid} domain',
                showarrow=False,
                y=1.1,
                font={'size': 18},
                )

    fig.update_layout({
        'title': 'Predictions',
        'height': 1200
        })

    return fig

def plot_cvs(fits):
    """Summarizes the CV procedure"""
    return go.Figure(
        data=[
            go.Scatter(
                x=model['cv_grid'],
                y=model['cv_rss'],
                mode='lines+markers',
                name=name
                )
            for name, model in fits.items()
            ],
        layout={
            'yaxis_title': 'Loss',
            'xaxis': {'title': 'Hyperparameter', 'type': 'log'},
            'title': 'Cross-validation tuning',
            'width': 900,
            'height': 800
            }
        )

@case
def heterogeneous_snr(_, case_name, report_path):
    """
    Regularized high-dimensional models with variation in SNR between variables
    """

    # Len1 is working dimension, len2 number of replicas
    len1, len2 = 200, 100

    rng = np.random.default_rng(4589)

    noise_class = rng.choice(2, size=len1).astype(np.bool)

    low_noise, high_noise = .1, 1.

    noise_sd = np.where(noise_class, high_noise, low_noise)

    train, test = gen_hyperv(len1, len2, noise_sd)

    # Train and eval models on subsets
    # We train on low, high and pool settings, and plot for low and high but not
    # pooled

    subsets = {
        'low': ~noise_class,
        'high': noise_class,
        'pooled': np.full_like(noise_class, True)
        }

    fits = {
        k: linear_shrinkage_cv.fit(train[subset])
        for k, subset in subsets.items()
        }

    # Fit regularized model once for each eval case with the preset prior var
    refits = {
        k_fit: {
            k_refit: linear_shrinkage.fit(
                train[subset],
                prior_var=fit['cv_best'])
            for k_refit, subset in subsets.items()
            if k_refit != 'pooled'
            }
       for k_fit, fit in fits.items()
    }

    predictions = {
        k_fit: {
            k_refit: linear_shrinkage.predict_loo(
                refit,
                test[subsets[k_refit], 0]
                )
            for k_refit, refit in _refits.items()
            }
        for k_fit, _refits in refits.items()
    }

    initial_plot = plot_data(train, noise_class)
    fit_plot = plot_fits(train, test[:, 0], subsets, predictions)

    with open(report_path, 'w', encoding='utf8') as stream:
        write_header(stream, case_name)
        write_fig(stream,
            initial_plot,
            fit_plot,
            plot_cvs(fits)
            )
        write_footer(stream)
