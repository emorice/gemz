"""
Clustered matrix t models
"""

import sklearn.cluster

from gemz.model import Model, ModelSpec, TransformedModel
from .transformations import (AddedConstantModel, PlugInModel, GroupScaledModel,
    get_training_data)
from .mt_sym import StdMatrixT

class PatchworkModel(TransformedModel):
    """
    Hard per-column combination of several inner models
    """
    def __init__(self, inner: Model, **params):
        super().__init__(inner)
        self.add_param('groups')
        self.bind_params(**params)

    def _condition(self, unobserved_indexes, data, **params):
        this_params, inner_params = self._split_params(params) # FIXME: dedup too
        groups = self.get_params(**this_params)['groups']

        raise NotImplementedError

class PreClusterModel(TransformedModel):
    """
    Mixin that clusters the observed data and makes the result available to the
    inner model.

    At present, this requires that the inner model accepts a parameter named
    "groups"
    """
    def __init__(self, inner, **params):
        super().__init__(inner)
        self.add_param('n_groups', None, None)
        self.bind_params(**params)

    def _condition(self, unobserved_indexes, data, **params):
        # 1 Extract "training" data
        # Select training data
        training_data = get_training_data(data, unobserved_indexes)

        # 2 Run k means
        # FIXME: this is duplicate with models.kmeans
        # Instead, we could have an interface for clustering models, and this
        # class would consume it.
        this_params, inner_params = self._split_params(params) # FIXME: dedup too
        n_groups = self.get_params(**this_params)['n_groups']
        sk_model = sklearn.cluster.KMeans(
                n_clusters=n_groups,
                n_init='auto',
                random_state=0, # FIXME: handle
                )

        sk_fit = sk_model.fit(training_data.T)
        groups = sk_fit.labels_

        # 3 Call inner model with the cluster results as a parameter
        return self.inner._condition(unobserved_indexes, data, groups=groups, **inner_params)

def make_model(spec: ModelSpec):
    """
    Model creation entry point
    """
    assert spec['model'] == 'clmt'
    return PreClusterModel(
        PlugInModel(
            PatchworkModel(
                GroupScaledModel(
                    AddedConstantModel(
                        StdMatrixT()
                        )
                    )
                )
            ),
        n_groups=spec['n_groups']
        )
