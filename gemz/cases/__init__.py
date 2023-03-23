"""
Module containing the demonstration cases
"""

import importlib
import logging
import pkgutil
import os
from typing import Callable, Type
from abc import ABC, abstractmethod

from gemz.models import ModelSpec

from .output import Output
from .output_html import HtmlOutput

class Case(ABC):
    """
    A case study meant to test and demonstrate how models behave on a specific
    dataset
    """
    name: str = '<case_name>'

    @abstractmethod
    def __call__(self, output: Output, model_specs: list[ModelSpec] | None = None) -> None:
        """
        Run the case study.

        By default all model_specs are used, but you can use the model_specs
        parameter to only run subsets of models or try new models not included
        in the default list.
        """

    @property
    @abstractmethod
    def model_specs(self) -> list[ModelSpec]:
        """
        Return a deafult list of model specs tested by the case
        """

_CaseFunction = Callable[[Output], None]

class CaseFunction(Case):
    """
    Wraps a function in a Case instance for compat.
    """
    def __init__(self, function: _CaseFunction):
        self._function = function
        self.__name__ = function.__name__

    def __call__(self, output: Output, model_specs: list[ModelSpec] | None = None) -> None:
        if model_specs not in (self.model_specs, None):
            raise NotImplementedError
        return self._function(output)

    @property
    def model_specs(self) -> list[ModelSpec]:
        """
        Return a default list of model specs tested by the case
        """
        # Symbolic model names ignored
        return [{'model': 'all'}]

CaseDef = _CaseFunction | Type[Case]

def case(case_def: CaseDef) -> CaseDef:
    """
    Decorator to register a demo case

    If a callable, kept as is.
    """

    if isinstance(case_def, type):
        # Instantiate class, purely because it is much more convenient to work
        # with trivial instances than classes
        _cases[case_def.name] = case_def()
    else: # Function
        case_name = case_def.__name__
        _cases[case_name] = CaseFunction(case_def)

    return case_def

def get_cases() -> dict[str, Case]:
    """
    Get the dictionary of existing case entry points
    """
    return _cases

def get_report_extension() -> str:
    """
    Get the extension to add to a case name to build its default report path
    """
    return '.html'

def get_report_path(output_dir: str, name: str) -> str:
    """
    Build the default report path for a case
    """
    ext = get_report_extension()
    return os.path.join(
        output_dir,
        name + ext
        )


_cases : dict[str, Case] = {}

_self_module = importlib.import_module(__name__)
for module_info in pkgutil.walk_packages(
    _self_module.__path__, _self_module.__name__ + '.'):
    importlib.import_module(module_info.name)
