"""
Low dimensional linearly distributed data with background noise
"""

import numpy as np
import plotly.express as px

from gemz.cases import Case, Output

class LinHet(Case):
    """
    Low dimensional linearly distributed data with background noise
    """
    name = 'lin_het'

    @property
    def model_specs(self):
        return [
                {'model': 'linear'}
                ]

    def gen_data(self, output: Output):
        low_dim = 2
        high_dim = 100

        spectrum = np.array([1., .1])

        rng = np.random.default_rng(0)

        ortho, _ = np.linalg.qr(rng.normal(size=(low_dim, low_dim)))

        innovations = rng.normal(size=(low_dim, high_dim))

        data = ((ortho * spectrum) @ ortho.T) @ innovations

        output.add_figure(px.scatter(x=data[0], y=data[1]))

        return data

    def run_model(self, spec, data):
        return None, None

    def _add_figures(self, output, data, spec, fit, preds):
        pass
