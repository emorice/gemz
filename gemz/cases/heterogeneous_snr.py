"""
Heterogenous signal-to-noise ratios
"""

import numpy as np

from plotly import graph_objects as go
from plotly.subplots import make_subplots

from gemz.cases import case
from gemz.cases.linear_reg import gen_hyperv
from gemz.reporting import write_header, write_footer, write_fig

from gemz import models

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

    fig = make_subplots(3, 2, shared_xaxes='all', shared_yaxes='all')

    for i_fit, (k_fit, fit_predictions) in enumerate(predictions.items()):
        for i_refit, k_refit in enumerate(['low', 'high']):
            subset = subsets[k_refit]
            imp_order = np.argsort(rev_order[subset])

            sub_covariate = covariate[subset][imp_order]

            coos = i_fit + 1, i_refit + 1
            fid = i_fit * 2 + i_refit
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

            prediction_displays = []
            if k_refit in fit_predictions:
                prediction_displays.append((
                    fit_predictions[k_refit].flatten()[imp_order],
                    'blue', 'Using only low or high'
                    ))
            if 'pooled' in fit_predictions:
                prediction_displays.append((
                    fit_predictions['pooled'].flatten()[subset][imp_order],
                    'red', 'Using all features'
                    ))

            for preds, color, name in prediction_displays:
                fig.add_trace(
                    go.Scatter(
                        x=sub_covariate,
                        y=preds,
                        mode='lines',
                        name=name,
                        legendgroup=name,
                        showlegend=not fid,
                        line={'color': color}
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

def eval_model(model, train, test, subsets):
    """
    Run models with one loss aggregation strategy
    """
    name, args = model

    module = models.get(name)

    fits = {
        k: module.fit(train[subset], **args)
        for k, subset in subsets.items()
        }

    # Fit regularized model once for each eval case with the preset prior var
    if name == 'linear_shrinkage_cv':
        refits = {
            k_fit: {
                k_refit: models.linear_shrinkage.fit(
                    train[subset],
                    prior_var=fit['cv_best'])
                for k_refit, subset in subsets.items()
                }
           for k_fit, fit in fits.items()
        }
        pred_module = models.linear_shrinkage
    else:
        # For diagonal target chimera models are not defined so just skip
        # re-fits
        refits = {
            k_fit: {
                k_fit: fit
                }
            for k_fit, fit in fits.items()
            }
        pred_module = module

    predictions = {
        k_fit: {
            k_refit: pred_module.predict_loo(
                refit,
                test[subsets[k_refit], 0]
                )
            for k_refit, refit in _refits.items()
            }
        for k_fit, _refits in refits.items()
    }

    fit_plot = plot_fits(train, test[:, 0], subsets, predictions)

    return [fit_plot, plot_cvs(fits)]

@case
def heterogeneous_snr(_, case_name, report_path):
    """
    Regularized high-dimensional models with variation in SNR between variables
    """

    # Len1 is working dimension, len2 number of replicas
    len1, len2 = 200, 100

    rng = np.random.default_rng(4589)

    noise_class = rng.choice(2, size=len1).astype(np.bool)

    noise_sd = np.where(noise_class, 1.0, 0.1)

    train, test = gen_hyperv(len1, len2, noise_sd)

    # Train and eval models on subsets
    # We train on low, high and pool settings, and plot for low and high but not
    # pooled

    subsets = {
        'low': ~noise_class,
        'high': noise_class,
        'pooled': np.full_like(noise_class, True)
        }

    initial_plot = plot_data(train, noise_class)

    model_definitions = [
        ('linear_shrinkage_cv', {'loss_name': 'RSS'}),
        ('linear_shrinkage_cv', {'loss_name': 'GEOM'}),
        ('lscv_precision_target', {})
        ]

    with open(report_path, 'w', encoding='utf8') as stream:
        write_header(stream, case_name)

        write_fig(stream, initial_plot)

        for model_def in model_definitions:
            name, args = model_def
            figs = eval_model(model_def, train, test, subsets)
            print("<h2>", name, *[f"{k} = {v}" for k,v in args.items()], "</h2>", file=stream)
            write_fig(stream, *figs)

        write_footer(stream)
