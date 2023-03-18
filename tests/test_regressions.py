"""
Output should be consistent between runs
"""

import json

import pytest

import gemz.cases
from gemz.cases import Output

class RegressionCaseOutput(Output):
    """
    Case output that compares outputs to previous runs

    Must be used as a context manager, and ONCE per test at most, with NO other
    use of pytest-regression. Pytest-regression generates one file per test, so
    successive uses will overwrite said file.

    Data is appended internally, then checked on exit.
    """
    def __init__(self, data_regression):
        self.data_regression = data_regression
        self.data = {'figures': []}
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def add_title(self, title: str):
        """
        Add title. Should be called at most once and first.
        """
        self.data['title'] = title

    def add_figure(self, figure):
        """
        Add figure to output
        """
        assert self.entered
        pure_dict = json.loads(figure.to_json())
        self.data['figures'].append(pure_dict)

    def __exit__(self, *args):
        """
        Checks regressions on closing
        """
        self.data_regression.check(self.data)
        self.entered = False

regression = pytest.mark.regression

@regression
@pytest.mark.parametrize('case', gemz.cases.get_cases().items(),
                         ids=lambda c: c[0])
def test_case(case, data_regression):
    """
    Run models on linear factor data
    """
    _case_name, case_function = case
    with RegressionCaseOutput(data_regression) as output_checker:
        case_function(output_checker)
