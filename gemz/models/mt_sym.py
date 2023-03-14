"""
Symmetric matrix-t model
"""

from numpy.typing import ArrayLike

from gemz.jax_utils import maximize, Bijector, Exp
from gemz.stats.matrixt import NonCentralMatrixT
from gemz.linalg import ScaledIdentity, Identity
from . import methods

class Method:
    """
    Prototype interface for methods
    """
    @classmethod
    def fit(cls, observed: ArrayLike):
        """
        Legacy wrapper for learn along the second axis
        """
        return cls.learn(observed, axis=1)

    @classmethod
    def learn(cls, observed: ArrayLike, axis: int):
        """
        Estimate model parameters in a way suitable for application along the
        given axis (that is, in a scenario where new data is appended to
        observed by concatenation along axis: 0 for new columns, 1 for new rows
        """
        raise NotImplementedError

class Model:
    """
    TBD
    """

class MaximumPseudoLikelihood(Method):
    """
    Mixin class to turn a multivariate distribution with free parameters into a
    predictive method
    """
    @classmethod
    def init_parameters(cls) -> dict[str, ArrayLike]:
        """
        Return reasonable values for all parameters
        """
        raise NotImplementedError

    @classmethod
    def get_bijectors(cls) -> dict[str, Bijector]:
        """
        Return reasonable bijectors for all free parameters
        """
        raise NotImplementedError

    @classmethod
    def make_model(cls, parameters: dict[str, ArrayLike], shape: tuple[int, int]) -> Model:
        """
        Build model from parameters
        """

    @classmethod
    def learn(cls, observed, axis: int):

        def _log_pseudolikelihood(**params: ArrayLike) -> float:
            return (cls
                    .make_model(params, observed.shape)
                    .pseudo_logpdf(observed)
                    )

        max_results = maximize(
            _log_pseudolikelihood,
            init=cls.init_parameters(),
            data={
                'observed': observed,
                },
            bijectors=cls.get_bijectors(),
            scipy_method='L-BFGS-B',
            )

        return (
            cls
            .make_model(max_results['opt'], observed.shape)
            .learn(observed, axis=axis)
            )


@methods.add('mt_sym')
class SymmetricMatrixT(MaximumPseudoLikelihood):
    """
    Predicitive model based on a matrix-t distribution with a symmetric
    parametrization
    """
    @classmethod
    def make_model(cls, parameters, shape: int) -> Model:
        params = parameters
        return NonCentralMatrixT.from_params(
                dfs=params['dfs'],
                left=ScaledIdentity(params['scale'], shape[0]),
                right=Identity(shape[1]),
                gram_mean_left=params['gram_mean_left'],
                gram_mean_right=params['gram_mean_right']
                )

    @classmethod
    def init_parameters(cls):
        return {
            'dfs': 1.,
            'scale': 1.,
            'gram_mean_left': 1.,
            'gram_mean_right': 1.,
            }

    @classmethod
    def get_bijectors(cls):
        return {
            'dfs': Exp(),
            'scale': Exp(),
            'gram_mean_left': Exp(),
            'gram_mean_right': Exp(),
            }
