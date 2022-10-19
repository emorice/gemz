"""
Plotting utils
"""

import pandas as pd
import plotly.graph_objects as go

from . import models

def plot_cv(spec, fit):
    """
    Generate CV summary figure
    """

    grid = fit['grid']
    losses, specs = zip(*(
            (result['loss'], spec)
            for spec, result in grid
            ))

    grid_axes = models.get(spec['inner']['model']).cv.get_grid_axes(specs)

    loss_df = pd.DataFrame(
            {axis['name']: axis['values'] for axis in grid_axes}
            ).assign(loss=losses)

    return [
        go.Figure(
            data=[
                go.Scatter(
                    x=loss_df[axis['name']],
                    y=loss_df['loss'],
                    mode='markers',
                    ),
                go.Scatter(
                    **(loss_df
                        .groupby(axis['name'], as_index=False)
                        ['loss']
                        .min()
                        .rename({'loss': 'y', axis['name']: 'x'}, axis=1)
                        ),
                    mode='lines',
                    )
                ],
            layout={
                'title': f'Cross-validated loss for {models.get_name(spec)}'
                    f' along {axis["name"]}',
                'xaxis': {
                    'title': axis['name'].capitalize(),
                    'type': 'log' if axis['log'] else 'linear'
                    },
                'yaxis': { 'title': fit['loss_name']},
                }
            )
        for axis in grid_axes
        ]
