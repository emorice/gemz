"""
Plotly-specific interfaces to model diagnostics
"""
import numpy as np
import plotly.graph_objects as go

class Plotly:

    @classmethod
    def optimization_trace(cls, values, name):
        return go.Figure([
            go.Scatter(
                x=np.arange(len(values)),
                y=values,
                mode='lines+markers'
                ),
            ], {
                'xaxis.title': 'Iterations',
                'yaxis.title': name,
                'title': 'Convergence'
                })
