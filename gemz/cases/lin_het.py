"""
Low dimensional linearly distributed data with background noise
"""

from typing import Iterator, Any
from itertools import product

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from gemz.cases import BaseCase, PerModelCase, Output
from gemz.models import ModelSpec
from gemz import Model

class LinHet(PerModelCase):
    """
    Low dimensional linearly distributed data with background noise
    """
    name = 'flex'

    def __init__(self):
        self.low_dim = 2
        self.high_dim = 101
        self.high_train = 50

        self.params = {
            'dims': {
                'low_dim': 2,
                #'high_dim': 100,
                },
            'heterogeneity': {
                'homo': False,
                #'het': True
                },
            'nonlinearity': {
                'linear': False,
                #'nonlinear': True
                },
            'model': self.model_unique_names
            }

    @property
    def model_specs(self):
        return [
                {'model': 'linear'},
                {'model': 'mt_sym'}
                ]


    def get_params(self) -> Iterator[tuple[str, Any]]:
        """
        Get the collections of case parameters  to try and the corresponding
        unique readable string.

        Default is to iterate over unique model specs.
        """
        param_triplets = (
                [ (param_name, value_printable, value) 
                    for value_printable, value in
                    values.items()]
                for param_name, values in self.params.items()
                )
        for triplets in product(*param_triplets):
            printable = ' x '.join(triplet[1] for triplet in triplets)
            yield f'{self.name} {printable}', triplets

    def gen_data(self, output: Output):
        spectrum = np.array([1., .1])

        rng = np.random.default_rng(0)

        ortho, _ = np.linalg.qr(rng.normal(size=(self.low_dim, self.low_dim)))

        innovations = rng.normal(size=(self.low_dim, self.high_dim))

        data = ((ortho * spectrum) @ ortho.T) @ innovations

        output.add_figure(px.scatter(x=data[0], y=data[1]))

        return data

    def run_model(self, spec, data):
        preds = (
                Model.from_spec(spec)
                .condition[1, self.high_train:](data)
                .mean
                )
        return None, preds

    def _add_figures(self, output: Output, data, spec: ModelSpec, fit, preds):
        output.add_figure(
                go.Figure(data=[
                    go.Scatter(x=data[0, self.high_train:], y=data[1, self.high_train:],
                        mode='markers', name='data'),
                    go.Scatter(x=data[0, self.high_train:], y=preds[0],
                        mode='markers', name='predictions')
                    ])
                )
