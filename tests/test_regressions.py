"""
Output should be consistent between runs
"""

import pytest

import gemz.cases
from gemz.cases import Output

class RegressionCaseOutput(Output):
    """
    Case output that compares outputs to previous runs
    """
    def add_title(self, title: str):
        """
        Add title. Should be called at most once and first.
        """
        del title

    def add_figure(self, figure):
        """
        Add figure to output
        """
        del figure

regression = pytest.mark.regression

@regression
@pytest.mark.parametrize('case', gemz.cases.get_cases().items(),
                         ids=lambda c: c[0])
def test_linear(case):
    """
    Run models on linear factor data
    """
    out = RegressionCaseOutput()
    case_name, case_function = case
    case_function(out)
