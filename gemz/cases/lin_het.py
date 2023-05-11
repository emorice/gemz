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
            'cond0': {
                'block': 'block',
                },
            'cond1': {
                'block': 'block',
                },
            'model': self.model_unique_names,
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


    def __call__(self, output: Output, case_params) -> None:
        _params_printable = { key: name for key, name, _ in case_params }
        params = { key: val for key, _, val in case_params }

        case_data = self.gen_data(output, case_params)

        spec = params['model']
        out = {}

        conditional =  (
            Model.from_spec(spec)
            .condition[1, self.high_train:](case_data)
            )
        out['conditional'] = conditional

        out['means'] = conditional.mean
        if hasattr(conditional, 'vars'):
            out['vars'] = conditional.vars
        else:
            out['vars'] = None

        self._add_figures(output, case_data, case_params, out)

    def gen_data(self, output: Output, case_params):
        """
        Generate a deterministic toy data set conforming to case_params
        """
        # TODO: actually honor case_params

        spectrum = np.array([1., .1])

        rng = np.random.default_rng(0)

        ortho, _ = np.linalg.qr(rng.normal(size=(self.low_dim, self.low_dim)))

        innovations = rng.normal(size=(self.low_dim, self.high_dim))

        data = ((ortho * spectrum) @ ortho.T) @ innovations

        output.add_figure(px.scatter(x=data[0], y=data[1]))

        return data

    def _add_figures(self, output: Output, data, params, model_out):
        output.add_figure(
                go.Figure(data=[
                    go.Scatter(x=data[0, self.high_train:], y=data[1, self.high_train:],
                        mode='markers', name='data'),
                    go.Scatter(x=data[0, self.high_train:], y=model_out['means'][0],
                        mode='markers', name='predictions')
                    ])
                )
