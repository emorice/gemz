"""
Abstract model interface
"""
from dataclasses import dataclass
import importlib

from gemz.models.methods import ModelSpec


Conditioner = tuple[int | slice, ...]

class Model:
    """
    Abstract model interface
    """
    def __init__(self, spec: ModelSpec, conditioner: Conditioner|None = None):
        self.spec = spec
        self.conditioner = conditioner

    @classmethod
    def from_spec(cls, spec: ModelSpec):
        """
        Instantiate model from a specification document

        The module defining the model is lazily loaded here.
        """
        if 'model' not in spec:
            raise ValueError('Invalid spec: spec must contain a "model" key')
        if spec['model'] == 'linear':
            module_name = 'gemz.models.linear'
        else:
            raise NotImplementedError(f'No such model: {spec["model"]}')

        module = importlib.import_module(module_name)

        return module.make_model(spec, None)

    @classmethod
    def from_conditionner(cls, spec: ModelSpec, unobserved: Conditioner):
        """
        Instantiate a model from a specification document and a model conditioner
        """
        return cls(spec, unobserved)

    @property
    def conditional(self):
        """
        Return an indexable object for convenient specification of conditional
        tasks
        """
        return ConditionMaker(self)

    def mean(self, *data, **kwdata):
        """
        Mean of unobserved data given observed data and parameters
        """
        raise NotImplementedError

@dataclass
class ConditionMaker:
    """
    Indexable object to easily condition a model
    """
    model: Model

    def __getitem__(self, keys):
        """
        Returns a conditionned copy of the original model
        """
        return self.model.from_conditionner(self.model, keys)
