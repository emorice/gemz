"""
Unified model interface
"""

import sys
from typing import TypedDict, Any

class ModelSpec(TypedDict):
    """
    Dictionnary containing the full specification of a model, in a way meant to
    be serialized, stored, hashed, etc. as needed.
    """
    model: str # model type identifier

# To be narrowed
_METHODS : dict[str, Any] = {}

def get(name: str):
    """
    Returns a model by name or None on unregistered model
    """
    return _METHODS.get(name)

def add(name: str):
    """
    Register a model class by name
    """
    def _set(cls):
        _METHODS[name] = cls
        return cls
    return _set

def add_module(name: str, module_name) -> None:
    """
    Register a model module by name
    """
    _METHODS[name] = sys.modules[module_name]

def get_name(spec: ModelSpec) -> str:
    """
    Return descriptive string from the spec
    """
    method = get(spec['model'])
    if hasattr(method, 'get_name'):
        return method.get_name(spec)
    return spec['model']
