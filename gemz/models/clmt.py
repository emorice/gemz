"""
Clustered matrix t models
"""

import numpy as np
import sklearn.cluster

from gemz.model import Model, ModelSpec, TransformedModel, Distribution
from .transformations import (AddedConstantModel, PlugInModel, GroupScaledModel,
    get_training_data, ScaledModel)
from .mt_sym import StdMatrixT

class PatchworkModel(TransformedModel):
    """
    Hard per-column combination of several inner models
    """
    def __init__(self, inner: Model, n_groups, **params):
        super().__init__(inner)
        self.n_groups = n_groups
        self.add_param('out_groups')
        self.bind_params(**params)

    def _condition(self, unobserved_indexes, data, **params):
        this_params, inner_params = self._split_params(params) # FIXME: dedup too
        groups = self.get_params(**this_params)['out_groups']

        # Loop or vectorize (??) over the inner models, then combine the results
        inner_params_template = self.inner.get_unbound_params()

        conditionals = []
        for i in range(self.n_groups):
            one_inner_params = {
                    key: inner_params[f'{key}_{i}']
                    for key in inner_params_template[0]
                    }
            conditionals.append(
                    self.inner._condition(unobserved_indexes, data,
                        **one_inner_params)
                    )

        return PatchworkDistribution(groups, conditionals)

    def get_unbound_params(self):
        this_params = self.get_local_unbound_params()

        # FIXME: vectorizing would undoubtedly be better, the main obstacle is
        # that we don't have an interface to vectorize the bijectors yet -- but
        # this should not be hard to add at some point.
        inner_params_template = self.inner.get_unbound_params()
        inner_params = [
                {
                    f'{key}_{copy}': value
                    for key, value in d.items()
                    for copy in range(self.n_groups)
                }
                for d in inner_params_template
                ]

        return [cur | inner  for cur, inner in zip(this_params, inner_params)]

class PatchworkDistribution(Distribution):
    """
    Hard combination of several underlying distributions
    """
    def __init__(self, groups, inners):
        self.inners = inners
        self.groups_p = groups

    def _aggregate(self, attr):
        result_xp = 0.
        for i, inner in enumerate(self.inners):
            result_xp = result_xp + getattr(inner, attr) * (self.groups_p == i)
        return result_xp

    @property
    def logpdf_observed(self):
        return self._aggregate('logpdf_observed')

    @property
    def mean(self):
        return self._aggregate('mean')

    @property
    def sf_radial_observed(self):
        return self._aggregate('sf_radial_observed')

class PreClusterModel(TransformedModel):
    """
    Mixin that clusters the observed data and makes the result available to the
    inner model.

    At present, this requires that the inner model accepts the groups as
    parameters which names are given in "target_params"
    "groups"
    """
    def __init__(self, inner, target_params, **params):
        super().__init__(inner)
        self.add_param('n_groups', None, None)
        self.bind_params(**params)
        self.target_params = target_params

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
        inner_params |= { name: groups for name in self.target_params }
        return self.inner._condition(unobserved_indexes, data, **inner_params)

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
                        ),
                    n_groups=spec['n_groups']
                    ),
                n_groups=spec['n_groups']
                )
            ),
        n_groups=spec['n_groups'],
        target_params=['in_groups', 'out_groups']
        )
