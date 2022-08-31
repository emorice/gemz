"""
Unified model interface
"""

import sys

_METHODS = {}

def get(name):
    """
    Returns a model by name
    """
    return _METHODS[name]

def add(name):
    """
    Register a model class by name
    """
    def _set(cls):
        _METHODS[name] = cls
        return cls
    return _set

def add_module(name, module_name):
    """
    Register a model module by name
    """
    _METHODS[name] = sys.modules[module_name]

def get_name(spec):
    """
    Return descriptive string from the spec
    """
    method = get(spec['model'])
    if hasattr(method, 'get_name'):
        return method.get_name(spec)
    return spec['model']
