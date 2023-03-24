"""
Default settings for plotly
"""

import os
import plotly
import plotly.graph_objects as go
import plotly.io as pio

template = go.layout.Template(layout={
        'width': 1000, 'height': 600, 'autosize': False,
        **{
            f'{a}axis': {
                'showline': True,
                'ticks': 'outside',
                'exponentformat': 'power',
                'constrain': 'domain'
                }
            for a in 'xy'
            },
        },
        data={
            'contour': [{'colorbar': {'exponentformat': 'power'}, 'opacity': 0.97}]
            }
    )
def plotly_init():
    pio.templates.default = template

def write(fig, name):
    os.makedirs('figs', exist_ok=True)
    params = {
        'width': 1000,
        'height': 1000,
        'scale': 1.,
        }
    pio.write_image(fig, f'figs/{name}.svg', 'svg', **params)
    pio.write_image(fig, f'figs/{name}.png', 'png', **params)
    return fig
