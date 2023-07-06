"""
Module containing the demonstration cases
"""

import importlib
import logging
import pkgutil
import os
from typing import Callable, Type, TypedDict, Any, Iterator, Generic, TypeVar
from collections import defaultdict
from itertools import product

from numpy.typing import ArrayLike

import gemz.models
from gemz.models import ModelSpec, get_name

from .output import Output
from .output_html import HtmlOutput

_cases : dict[str, 'BaseCase'] = {}

CaseParams = TypeVar('CaseParams')

class BaseCase(Generic[CaseParams]):
    """
    Base class for the case mechanism

    Due to to the prototyping process, this ended up with two interfaces to
    manipulate case paramaters:
     * get_params list combination of parmaeters to try in the test harness
     * parameters, get_param_combinations, get_display_id list combination of
     parameters to browse the result.

    The main difference is that the first deals with native python objects to
    pass to the model, while the others mostly export strings for interfaces.
    """
    name: str = ''

    def __call__(self, output: Output, case_params: CaseParams) -> None:
        raise NotImplementedError

    def __init_subclass__(cls, /, abstract_case=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if abstract_case:
            return

        if not cls.name or cls.name in _cases:
            raise ValueError(f'{cls.name} already registered, give the Case a'
                    ' new unique name attribute')
        _cases[cls.name] = cls()

    @property
    def parameters(self):
        """
        Return a dictionary of valid parameter names for the case, mapped to
        list of corresponding printable values.
        """
        return {}

    def get_param_combinations(self):
        """
        Cartesian product of the values in parameters
        """
        param_matrix = [
                [ {
                    'name': param_name,
                    'display_name': param_def['display_name'],
                    'value': value
                    }
                    for value in param_def['values']
                    ]
                for param_name, param_def in self.parameters.items()
                ]
        for combination in product(*param_matrix):
            yield combination, self.get_display_id({
                param['name']: param['value']
                for param in combination
                })

    def get_display_id(self, params):
        """
        Generate a consistent, unambiguous, printable string for each
        combination of parameters
        """
        return ' x '.join(params[name] for name in self.parameters)

    def get_params(self) -> Iterator[tuple[str, CaseParams]]:
        """
        Get the collections of case parameters  to try and the corresponding
        unique readable string.

        Note: this should be more properly called get_params_ *combinations*,
        we'll fix that eventually
        """
        raise NotImplementedError

class PerModelCase(BaseCase[CaseParams], abstract_case=True):
    """
    Case parametrized by models
    """
    @property
    def model_specs(self) -> list[ModelSpec]:
        """
        Return a default list of model specs tested by the case
        """
        raise NotImplementedError

    @property
    def model_unique_names(self) -> dict[str, ModelSpec]:
        """
        Unique model names
        """
        duplicates: dict[str, int] = defaultdict(int)
        for spec in self.model_specs:
            duplicates[get_name(spec)] += 1
        ids : dict[str, int] = defaultdict(int)
        unique_names = {}
        for spec in self.model_specs:
            name = get_name(spec)
            unique_names[
                    name + f'_{ids[name]}' if duplicates[name] > 1
                    else name
                    ] = spec
            ids[name] += 1
        return unique_names

class Case(PerModelCase[list[ModelSpec]], abstract_case=True):
    """
    A case study meant to test and demonstrate how models behave on a specific
    dataset
    """
    def __call__(self, output: Output, case_params: list[ModelSpec] | None = None) -> None:
        """
        Run the case study.

        By default all model_specs are used, but you can use the model_specs
        parameter to only run subsets of models or try new models not included
        in the default list.
        """
        if case_params is None:
            model_specs = self.model_specs
        else:
            model_specs = case_params

        data = self.gen_data(output)

        for spec in model_specs:
            fit, preds = self.run_model(spec, data)
            self._add_figures(output, data, spec, fit, preds)

    @property
    def parameters(self):
        return {
                'model': {
                    'display_name': 'Model',
                    'values': self.model_unique_names
                    }
                }

    def get_display_id(self, params):
        """
        Compat, we unfortunately used inconsistent naming in the past.
        """
        return 'x_' + super().get_display_id(params)

    def get_params(self) -> Iterator[tuple[str, Any]]:
        """
        Get the collections of case parameters  to try and the corresponding
        unique readable string.

        Default is to iterate over unique model specs.
        """
        for model_name, model_spec in self.model_unique_names.items():
            yield f'{self.name} x {model_name}', [model_spec]

    def run_model(self, spec: ModelSpec, data) -> tuple[Any, Any]:
        """
        Build model from spec, apply it to data and generate test plots.

        Returns:
            Two loosely defined objects, the "model fit", which is expected to
            contain data of a different type from one model to an other
            (typically parameter values, optimization traces...), and the "model
            prediction", which is the output of the model that is intended to be
            standardized and comparable from one model to an other.

            The actual types of these objects therefore depends on the model
            used and task at hand.
        """
        fit = gemz.models.fit(spec, data['train'])
        preds = gemz.models.predict_loo(spec, fit, data['test'])
        return fit, preds

    def gen_data(self, output: Output):
        """
        Data generation
        """
        raise NotImplementedError

    def _add_figures(self, output: Output, data, spec, fit, preds) -> None:
        """
        Regression figures generation
        """
        raise NotImplementedError


_CaseFunction = Callable[[Output], None]

class CaseFunction(BaseCase[None], abstract_case=True):
    """
    Wraps a function in a Case instance for compat.
    """
    def __init__(self, function: _CaseFunction):
        self._function = function
        self.name = function.__name__

    def get_params(self) -> Iterator[tuple[str, None]]:
        """
        Get the collections of case parameters to try and the corresponding
        unique readable string.

        In this case this is a single empty parameter sets since case functions
        are not parametrizable.
        """
        yield self.name, None

    def __call__(self, output: Output, params: None = None) -> None:
        if params is not None:
            raise NotImplementedError
        return self._function(output)

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

def get_cases() -> dict[str, BaseCase]:
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
