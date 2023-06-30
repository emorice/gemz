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

    @classmethod
    def optimized_parameters(cls, init, final):
        return [
                go.Figure([
                    go.Bar(
                        x=['Initial', 'Final'],
                        y=[init[param_name], param_values],
                        )
                    ],
                    {
                        'title': f'Optimized parameter {param_name}'
                        }
                    )
                for param_name, param_values in final.items()
                ]
