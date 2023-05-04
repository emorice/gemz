"""
Output should be consistent between runs
"""

import json
from typing import Iterator, Any

import pytest

import gemz.cases
import gemz.models
from gemz.models import ModelSpec
from gemz.cases import Output, Case

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

    def __exit__(self, exc_type, _exc_value, _traceback):
        """
        Checks regressions on closing
        """
        if exc_type is None:
            # Do not check data on error, because this would improperly create a
            # -- likely invalid ! -- regression file even if the test did not run
            # til the end
            self.data_regression.check(self.data)
        self.entered = False

regression = pytest.mark.regression

def pytest_cases() -> Iterator[Any]:
    """
    Iterator over case studies
    """
    for _name, case in gemz.cases.get_cases().items():
        for case_id, case_params in case.get_params():
            yield pytest.param(case, case_params, id=case_id)

@regression
@pytest.mark.parametrize('case, params', pytest_cases())
def test_case(case: Case, params, data_regression):
    """
    Run models on linear factor data
    """
    with RegressionCaseOutput(data_regression) as output_checker:
        case(output_checker, params)
