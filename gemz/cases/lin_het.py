"""
Low dimensional linearly distributed data with background noise
"""

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
