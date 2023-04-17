"""
Low dimensional linearly distributed data with background noise
"""

import numpy as np
import plotly.express as px

from gemz.cases import Case, Output
from gemz import Model

class LinHet(Case):
    """
    Low dimensional linearly distributed data with background noise
    """
    name = 'lin_het'

    def __init__(self):
        self.low_dim = 2
        self.high_dim = 100
        self.high_train = 50

    @property
    def model_specs(self):
        return [
                {'model': 'linear'}
                ]

    def gen_data(self, output: Output):
        spectrum = np.array([1., .1])

        rng = np.random.default_rng(0)

        ortho, _ = np.linalg.qr(rng.normal(size=(self.low_dim, self.low_dim)))

        innovations = rng.normal(size=(self.low_dim, self.high_dim))

        data = ((ortho * spectrum) @ ortho.T) @ innovations

        output.add_figure(px.scatter(x=data[0], y=data[1]))

        return data

    def run_model(self, spec, data):
        preds = Model.from_spec(spec).conditional[1, self.high_train:].mean(data)
        return None, preds

    def _add_figures(self, output, data, spec, fit, preds):
        pass
