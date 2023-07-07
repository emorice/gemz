"""
Low dimensional linearly distributed data with background noise
"""

from typing import Iterator, Any
from itertools import product

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from gemz import diagnostics
from gemz.cases import PerModelCase, Output
from gemz.model import EachIndex, Model, IndexLike

class LinHet(PerModelCase):
    """
    Low dimensional linearly distributed data with background noise
    """
    name = 'flex'

    def __init__(self):
        low_dim = 2
        self.high_dim = 101

        self.params = {
            'dims': {
                'low_dim': 2,
                'high_dim': 95,
                },
            'centered': {
                'centered': True,
                'noncentered': False,
                },
            'heterogeneity': {
                'homo': False,
                'het': True
                },
            'nonlinearity': {
                'linear': False,
                'nonlinear': True
                },
            'cond0': {
                'block': 'block-one',
                },
            'cond1': {
                'block': self.make_index('block', self.high_dim),
                'loo'  : self.make_index('loo', self.high_dim),
                },
            'model': self.model_unique_names,
            }

    @property
    def parameters(self):
        return { name: {
                    'display_name': name,
                    'values': values
                    }
                for name, values in self.params.items()
                }

    @property
    def model_specs(self):
        return [
                {'model': 'linear'},
                {'model': 'mt_std'},
                {'model': 'mt_sym', 'scale': 'auto'},
                {'model': 'mt_het'},
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


    def make_index(self, param: str, length: int) -> IndexLike:
        """
        Create an index-like object matching the param string: either a slice
        to the second half of the axis if 'block' or EachIndex for 'loo'
        """
        if param == 'block':
            return slice(length // 2, None)
        if param == 'block-one':
            return slice(length - 1, None)
        if param == 'loo':
            return EachIndex
        raise NotImplementedError(param)

    def __call__(self, output: Output, case_params) -> None:
        _params_printable = { key: name for key, name, _ in case_params }
        params = { key: val for key, _, val in case_params }

        case_data = self.gen_data(output, case_params)

        spec = params['model']
        out = {}

        cond0 = self.make_index(params['cond0'], params['dims'])
        cond1 = params['cond1']

        conditional =  (
            Model.from_spec(spec)
            .conditional[cond0, cond1](case_data[0])
            )
        out['conditional'] = conditional

        # Extract all the constant statistics
        # Not sure this will make it to the final interface
        out = conditional.as_dict()

        self._add_figures(output, case_data, case_params, out)

        # Diagnostic plots
        output.add_figures(
            conditional.export_diagnostics(diagnostics.Plotly)
            )

    def gen_data(self, output: Output, case_params):
        """
        Generate a deterministic toy data set conforming to case_params
        """
        # todo: dedup
        params = { key: val for key, _printable, val in case_params }
        low_dim = params['dims']

        num_classes = 1
        num_good_classes = 1
        # TODO: honor all case_params
        if params['heterogeneity']:
            num_classes += 1

        if params['nonlinearity']:
            num_classes += 1
            num_good_classes += 1

        spectrum = np.array(
                ([1.] * (low_dim // 2)) + ([.1] * (low_dim - low_dim // 2))
                )

        rng = np.random.default_rng(0)

        # Affect random classes
        classes_p = rng.choice(num_classes, size=self.high_dim)

        # Draw covariance and mean for each non-noise class
        ortho_knn, _ = np.linalg.qr(rng.normal(size=(num_good_classes, low_dim, low_dim)))

        # Draw a mean for everyone, including noise class
        if not params['centered']:
            means_kn1 = rng.normal(size=(num_classes, low_dim, 1))
        else:
            means_kn1 = np.zeros((1, 1, 1))

        innovations_np = rng.normal(size=(low_dim, self.high_dim))

        innovations_knp = innovations_np + 3.* means_kn1

        transformations_knn = (ortho_knn * spectrum) @ np.swapaxes(ortho_knn, -1, -2)
        if params['heterogeneity']:
            transformations_knn = np.concatenate((
                transformations_knn,
                np.eye(low_dim)[None, ...]
                ))

        data_knp = transformations_knn @ innovations_knp

        data_np = np.take_along_axis(data_knp, classes_p[None, None, :], 0)[0]

        output.add_figure(px.scatter(x=data_np[0], y=data_np[-1],
            title='Dataset (first and last row)').update_traces(
                marker={'symbol': classes_p}))

        return data_np, classes_p

    def make_union_slice(self, index: IndexLike):
        """
        Convert index to a slice covering all values appearing in index
        """
        if isinstance(index, slice):
            return index
        if index is EachIndex:
            return slice(None)
        raise NotImplementedError

    def _add_figures(self, output: Output, case_data, case_params, model_out):
        data, classes = case_data
        # todo: dedup
        params = { key: val for key, _, val in case_params }

        is_test_col = self.make_union_slice(params['cond1'])

        truth_x = data[0, is_test_col]
        truth_y = data[-1, is_test_col]
        classes = classes[is_test_col]

        # means is a 1 x n_test_columns matrix
        pred_y = model_out['mean'][0]

        output.add_figure(
                go.Figure(data=[
                    go.Scatter(x=truth_x, y=truth_y, mode='markers',
                        name='data', marker={'symbol': classes}),
                    go.Scatter(x=truth_x, y=pred_y, mode='markers',
                        name='predictions', marker={'symbol': classes}),
                    ],
                    layout={
                        'title': 'Predicted mean and ground truth',
                        'yaxis.title': 'Predicted value',
                        'xaxis.title': 'First predictor'
                        })
                )

        output.add_figure(
                go.Figure(data=[
                    go.Scatter(x=pred_y, y=pred_y, name='Expected'),
                    go.Scatter(x=pred_y, y=truth_y, mode='markers',
                        name='Predicted', marker={'symbol': classes}),
                    ],
                    layout={
                        'title': 'Predicted mean vs ground truth',
                        'xaxis.title': 'Predicted values',
                        'yaxis.title': 'Orignal values',
                        })
                )

        survf = model_out['sf_radial']
        if survf is not None:
            survf = np.sort(survf)
            fraction = np.arange(len(survf)) / len(survf)

            # Calibration
            output.add_figure(
                    go.Figure(data=[
                            go.Scatter(x=survf, y=survf,
                                name='Expected uniform'),
                            go.Scatter(x=survf, y=fraction,
                                name='Observed survival function'),
                        ],
                        layout={
                            'title': 'Calibration',
                            'xaxis.title': 'Survival probability',
                            'yaxis.title': 'Cumulative fraction of observations',
                            })
                    )
