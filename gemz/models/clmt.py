"""
Clustered matrix t models
"""

from gemz.model import ModelSpec, TransformedModel
from .transformations import AddedConstantModel, PlugInModel, GroupScaledModel
from .mt_sym import StdMatrixT

class PatchworkModel(TransformedModel):
    """
    Hard per-column combination of several inner models
    """

class PreClusterModel(TransformedModel):
    """
    Mixin that clusters the observed data and makes the result available to the
    inner model
    """

def make_model(spec: ModelSpec):
    """
    Model creation entry point
    """
    assert spec == {'model': 'clmt'}
    return PreClusterModel(
        PlugInModel(
            PatchworkModel(
                GroupScaledModel(
                    AddedConstantModel(
                        StdMatrixT()
                        )
                    )
                )
            )
        )
