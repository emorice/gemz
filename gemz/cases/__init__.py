"""
Module containing the demonstration cases
"""

import importlib
import logging
import pkgutil
import os
from typing import Callable, Type
from abc import ABC, abstractmethod
from collections import defaultdict

from gemz.models import ModelSpec, get_name

from .output import Output
from .output_html import HtmlOutput

_cases : dict[str, 'Case'] = {}

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

    @property
    def model_unique_names(self):
        """
        Unique model names
        """
        duplicates = defaultdict(int)
        for spec in self.model_specs:
            duplicates[get_name(spec)] += 1
        ids = defaultdict(int)
        unique_names = []
        for spec in self.model_specs:
            name = get_name(spec)
            unique_names.append(
                    name + f'_{ids[name]}' if duplicates[name] > 1
                    else name
                    )
            ids[name] += 1
        return unique_names

    def __init_subclass__(cls, /, abstract_case=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if abstract_case:
            return
        if cls.name in _cases:
            raise ValueError(f'{cls.name} already registered, give the Case a'
                    ' new unique name attribute')
        _cases[cls.name] = cls()

_CaseFunction = Callable[[Output], None]

class CaseFunction(Case, abstract_case=True):
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

    @property
    def model_unique_names(self):
        return ['all']

CaseDef = _CaseFunction | Type[Case]

def case(case_def: CaseDef) -> CaseDef:
    """
    Decorator to register a demo case

    If a callable, kept as is.
    """

    assert not isinstance(case_def, type)

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


_self_module = importlib.import_module(__name__)
for module_info in pkgutil.walk_packages(
    _self_module.__path__, _self_module.__name__ + '.'):
    importlib.import_module(module_info.name)
