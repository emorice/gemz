"""
Adapted from the pytest doc
"""

import pytest

def pytest_addoption(parser):
    """
    Add regression selection option
    """
    parser.addoption(
        "--regression", action="store_true", default=False,
        help="run regression tests"
    )

def pytest_collection_modifyitems(config, items):
    """
    Convert regression marks to skips
    """
    if config.getoption("--regression"):
        return
    skip_reg = pytest.mark.skip(
            reason="Regression tests only run with --regression")
    for item in items:
        if "regression" in item.keywords:
            item.add_marker(skip_reg)
