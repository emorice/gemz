"""
Gaussian mixtrue based predictions.
"""

import numpy as np
import sklearn.mixture

from gemz import linalg
from .methods import add_module
from . import cv

add_module('gmm', __name__)

def fit(data, n_groups, bayesian=False, **skargs):
    """
    Compute clusters, cluster means and dispersion on given data
    """
    # data: len1 x len2, len2 is the one to split

    if bayesian:
        Mixture = sklearn.mixture.BayesianGaussianMixture
    else:
        Mixture = sklearn.mixture.GaussianMixture
    sk_model = Mixture(
        n_components=n_groups,
        covariance_type='full',
        random_state=1,
        **skargs,
        )

    sk_fit = sk_model.fit(data.T)

    # len2
    groups = sk_fit.predict(data.T)
    probas = sk_fit.predict_proba(data.T)

    # Sklearn applies a default reg if not given and we need it again at
    # prediction time
    reg_covar = skargs['reg_covar'] if 'reg_covar' in skargs else 1e-6

    model_min = {
        'data': data,
        'groups': groups,
        'responsibilities': (probas / probas.sum(-1, keepdims=True)).T,
        'reg_covar': reg_covar,
        }

    return model_min

def predict_loo(model, new_data):
    """
    Args:
        new_data: len1' x len2, with len2 matching training data
    """
    # Index names:
    #  n: len1, small dim
    #  p: len2, large dim
    #  m: len1', new small dim
    #  k: groups

    new_data_mp = new_data
    data_np = model['data']
    resps_kp = model['responsibilities']

    preds_pm = np.zeros_like(new_data_mp.T)

    for resps_p, reg_covar in zip(resps_kp,
            model['reg_covar'] * np.ones(len(resps_kp))
            ):

        model_one = precompute_loo(dict(
            data=data_np,
            responsibilities=resps_p,
            reg_covar=reg_covar))


        group_sizes_p = model_one['group_sizes']
        means_pn = model_one['means']
        precisions_pnn = model_one['precisions']

        new_means_pm = (
            linalg.loo_matmul(resps_p, new_data_mp.T)
            / group_sizes_p[..., None]
            )

        new_grams_pmn = (
            linalg.loo_cross_square(new_data_mp, resps_p, data_np)
            / group_sizes_p[..., None, None]
        )

        new_covariances_pmn = linalg.LowRankUpdate(
            new_grams_pmn,
            # Dummy contraction, last
            new_means_pm[..., None], # pm1
            -1,
            # Dummy contraction, 2nd to last
            means_pn[..., None, :] # p1n
            )

        centered_data_pn = data_np.T - means_pn

        trans_data_pn1 = precisions_pnn @ centered_data_pn[..., None]

        preds_one_pm = (
            new_means_pm +
            (new_covariances_pmn @ trans_data_pn1)[..., 0] # pmn pn1 -> pm1
            )

        preds_pm += preds_one_pm * resps_p[..., None]

    return preds_pm.T

cv = cv.CartesianCV(
        cv.Int1dCV('n_groups', 'components'),
        cv.Real1dCV('reg_covar', 'prior variance')
        )

def get_name(spec):
    """
    Descriptive name
    """
    name = f"{spec['model']}/{spec['n_groups']}"

    if 'reg_covar' in spec:
        name = f'{name}:{spec["reg_covar"]}'

    return name

# Main computations
# =================

def precompute(model):
    """
    Generate a model with useful precomputed quantities from minimal spec
    """
    data = model['data']
    responsibilities = model['responsibilities']

    # K, sum over P
    group_sizes = np.sum(responsibilities, -1)

    # K x N, sum over P
    means = responsibilities @ data.T / group_sizes[:, None]
    # K x N x N. O(KNP) intermediate, may be improved
    covariances = (
        (data * responsibilities[:, None, :]) @ data.T
            / group_sizes[:, None, None]
        - means[:, :, None] * means[:, None, :]
        )
    # K x N X N
    precisions = np.linalg.inv(covariances)

    return {
        **model,
        'group_sizes': group_sizes,
        'means': means,
        'precisions': precisions
        }

def precompute_loo(model):
    """
    Generate a model with useful LOO precomputed quantities from minimal spec
    """
    data_np = model['data']
    resps_kp = model['responsibilities']

    # If no reg was needed at fit time we assume none at predict time
    reg_covar = model['reg_covar'] if 'reg_covar' in model else 0.

    group_sizes_kp = linalg.loo_sum(resps_kp, -1)
    group_sizes_k = np.sum(resps_kp, -1)

    # Regularize sligthly the group sizes to avoid nans coming from empty groups
    # Must be a bit bigger than an eps, but much smaller than 1
    # Sklearn uses 10 eps of the dtype
    group_sizes_kp += 1e-6

    means_kpn = linalg.loo_matmul(resps_kp, data_np.T) / (
            group_sizes_kp[..., None]
            )

    grams_kpnn = (
        linalg.loo_square(data_np, resps_kp, reg_b=reg_covar * group_sizes_k)
        / group_sizes_kp[..., None, None]
        )

    covariances_kpnn = linalg.SymmetricLowRankUpdate(
        base=grams_kpnn,
        # Extra contracting dim (outer product)
        factor=means_kpn[..., None],
        weight=-1,
        )

    precisions_kpnn = np.linalg.inv(covariances_kpnn)

    return {
        **model,
        'group_sizes': group_sizes_kp,
        'means': means_kpn,
        'covariances': covariances_kpnn,
        'precisions': precisions_kpnn
        }
