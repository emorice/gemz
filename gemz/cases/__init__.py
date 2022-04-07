"""
Module containing the demonstration cases
"""

import importlib
import logging
import pkgutil
import os

def case(function):
    """
    Decorator to register a demo case
    """

    key = function.__name__
    _cases[key] = function

    return function

def get_cases():
    """
    Get the dictionary of existing case entry points
    """
    return _cases

def get_report_extension():
    """
    Get the extension to add to a case name to build its default report path
    """
    return '.html'

def get_report_path(output_dir, name):
    """
    Build the default report path for a case
    """
    ext = get_report_extension()
    return os.path.join(
        output_dir,
        name + ext
        )

_cases = {}

_self_module = importlib.import_module(__name__)
for module_info in pkgutil.walk_packages(
    _self_module.__path__, _self_module.__name__ + '.'):
    importlib.import_module(module_info.name)
