"""
Plotting utils
"""

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

    grid_axis = models.get(spec['inner']['model']).get_grid_axis(specs)

    return go.Figure(
        data=[
            go.Scatter(
                x=grid_axis['values'],
                y=losses,
                mode='lines+markers',
                )
            ],
        layout={
            'title': f'Cross-validated loss for {models.get_name(spec)}',
            'xaxis': {
                'title': grid_axis['name'].capitalize(),
                'type': 'log' if grid_axis['log'] else 'linear'
                },
            'yaxis': { 'title': fit['loss_name']},
            }
        )
