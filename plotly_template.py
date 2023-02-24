"""
Default settings for plotly
"""

import plotly
import plotly.graph_objects as go

plotly.io.templates.default = go.layout.Template(layout={
    'width': 600, 'height': 600, 'autosize': False,
    **{
        f'{a}axis': {
            'showline': True,
            'ticks': 'outside',
            'exponentformat': 'power'
            }
        for a in 'xy'
        },
    },
    data={
        'contour': [{'colorbar': {'exponentformat': 'power'}, 'opacity': 0.97}]
        }
)
