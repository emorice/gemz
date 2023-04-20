"""
Symmetric matrix-t model
"""

import logging

import numpy as np
from numpy.typing import ArrayLike

from gemz.jax_utils import maximize, Bijector, Exp
from gemz.stats.matrixt import NonCentralMatrixT
from gemz.jax.linalg import ScaledIdentity
from gemz.model import ModelSpec, Conditioner, Model
from . import methods

class Method:
    """
    Prototype interface for methods
    """
    @classmethod
    def fit(cls, observed: ArrayLike, **options):
        """
        Legacy wrapper for learn along the second axis
        """
        return cls.learn(observed, options, axis=1)

    @classmethod
    def learn(cls, observed: ArrayLike, options, axis: int):
        """
        Estimate model parameters in a way suitable for application along the
        given axis (that is, in a scenario where new data is appended to
        observed by concatenation along axis: 0 for new columns, 1 for new rows
        """
        raise NotImplementedError

class MaximumPseudoLikelihood(Method):
    """
    Mixin class to turn a multivariate distribution with free parameters into a
    predictive method
    """
    @classmethod
    def init_parameters(cls, options: dict) -> dict[str, ArrayLike]:
        """
        Return reasonable values for all parameters
        """
        raise NotImplementedError

    @classmethod
    def get_bijectors(cls, otions: dict) -> dict[str, Bijector]:
        """
        Return reasonable bijectors for all free parameters
        """
        raise NotImplementedError

    @classmethod
    def make_model(cls, parameters: dict[str, ArrayLike],
            options: dict, shape: tuple[int, ...]):
        """
        Build model from parameters
        """
        raise NotImplementedError

    @classmethod
    def learn(cls, observed: ArrayLike, options: dict, axis: int):

        def _log_pseudolikelihood(**params: ArrayLike) -> float:
            return (cls
                    .make_model(params, options, np.shape(observed))
                    .pseudo_logpdf(observed)
                    )

        max_results = maximize(
            _log_pseudolikelihood,
            init=cls.init_parameters(options),
            data={
                'observed': observed,
                },
            bijectors=cls.get_bijectors(options),
            scipy_method='L-BFGS-B',
            )

        logging.info('%s: optimum %s', cls.__name__, max_results['opt'])

        return {
            'learned': (cls
                .make_model(max_results['opt'], options, np.shape(observed))
                .learn(observed, axis=axis)
                ),
            'opt': max_results
            }


@methods.add('mt_sym')
class SymmetricMatrixT(MaximumPseudoLikelihood):
    """
    Predicitive model based on a matrix-t distribution with a symmetric
    parametrization
    """
    @classmethod
    def make_model(cls, parameters, options, shape: tuple[int, ...]):
        params = parameters
        return NonCentralMatrixT.from_params(
                dfs=params['dfs'],
                left=ScaledIdentity(1., shape[0]),
                right=ScaledIdentity(params['scale'], shape[1]),
                gram_mean_left=params.get('gram_mean_left'),
                gram_mean_right=params.get('gram_mean_right')
                )

    @classmethod
    def init_parameters(cls, options):
        init_params = {
            'dfs': 1.,
            'scale': 1.,
            }
        if not ('centered' in options and options['centered'][0]):
            init_params['gram_mean_left'] = 1.
        if not ('centered' in options and options['centered'][1]):
            init_params['gram_mean_right'] = 1.
        return init_params

    @classmethod
    def get_bijectors(cls, options):
        bijs = {
            'dfs': Exp(),
            'scale': Exp(),
            }
        if not ('centered' in options and options['centered'][0]):
            bijs['gram_mean_left'] = Exp()
        if not ('centered' in options and options['centered'][1]):
            bijs['gram_mean_right'] = Exp()
        return bijs
    @classmethod
    def predict_loo(cls, learned, new_observed: ArrayLike):
        """
        Legacy wrapper for apply along axis 0
        """
        new_dims = new_observed.shape[0]
        new_gram = ScaledIdentity(1., new_dims)
        return learned['learned'].extend(new_gram=new_gram, axis=1).uni_cond(new_observed)[0]

# Interface V2

class SymmetricMatrixT2(Model):
    def mean(self, data):
        return np.zeros_like(self.conditioner.select(data))

def make_model(spec: ModelSpec, conditioner: Conditioner):
    return SymmetricMatrixT2(spec, conditioner)
