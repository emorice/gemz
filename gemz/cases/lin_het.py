"""
Low dimensional linearly distributed data with background noise
"""

import numpy as np

from gemz.cases import Case

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

    def gen_data(self, output):
        return { 'train': np.zeros((1,1)), 'test': np.zeros((1,1)) }

    def _add_figures(self, output, data, spec, fit, preds):
        pass
