"""
Clustered McKay model

Reformatted from experimental notebooks, hence the subpar coding style.
"""

import numpy as np
import sklearn.cluster
import jax
import jax.numpy as jnp
from jax import lax
from tqdm.auto import tqdm

from . import methods, cv as _cv

# High-level interface
# ====================

methods.add_module('cmk', __name__)

def fit(data, n_groups: int, n_iter: int = 100, verbose=True, jit=True) -> dict:
    """
    Fit the model with a fixed number of MK-updates iterations

    Args:
        data: N1 x N2. Creates a model of N2-dimensional loo problem from N1
            replicates.
    """

    # Initialize the model. `data` contains the fixed parts, `state` the variable
    # parts.
    # The cmk_* functions follow the replicate-first convention
    cmk_data, state = cmk_init(data, n_groups)

    # Duplicate to avoid accidentally modifying in-place
    hist = []
    abort = False
    abort_msgs = []
    iters = range(n_iter)
    if verbose:
        iters = tqdm(range(n_iter), desc=f'cmk/{n_groups}')
    if jit:
        cmk_many_fun = cmk_many_jit
    else:
        cmk_many_fun = cmk_many
    for i in iters:
        inter, aux = cmk_many_fun(**cmk_data, **state)
        updates = cmk_update(**cmk_data, **state, **inter, **aux)
        hist.append({
            'iteration': i,
            'log_likelihood': inter['log_evidence'].sum(),
            'cc_zeros': updates['vanished_compact_covariance']
        })
        if updates['inf_data_vars']:
            abort_msgs.append('Diverging data vars, aborting')
            abort = True
        if updates['inf_compact_covariance']:
            abort_msgs.append('Diverging compact covariance, aborting')
            abort = True
        for k in ['data_vars', 'compact_covariance']:
            if jnp.any(jnp.isnan(updates[f'new_{k}'])):
                abort_msgs.append(f'{k}: Nan, aborting')
                abort = True
        if abort:
            break
        for k in ['data_vars', 'compact_covariance']:
            state[k] = updates[f'new_{k}']
    return {
        'data': cmk_data,
        'state': state,
        'hist': hist,
        'aborted': abort,
        'errors': abort_msgs
    }

def predict_loo(model, new_data):
    """
    Leave-one-out prediction on new data

    Args:
        new_data: N3 x N2, with N2 matching the data given to `fit`.
    """
    final_inter, final_aux = cmk_many(**model['data'], **model['state'])
    # Once again, cmk_* functions follow replicate-first convention
    predictions = cmk_predict(
        new_data=new_data,
        **model['data'], **model['state'],
        **final_inter, **final_aux)

    return predictions

def get_name(spec):
    """
    Readable name
    """
    return f"{spec['model']}/{spec['n_groups']}"


cv = _cv.Int1dCV('n_groups', 'groups')

# CMK algorithm
# =============

def cmk_init(data, n_groups):
    """
    Returns constant pre-computed values (data) and initial values of the variable state.
    """
    clusters = sklearn.cluster.KMeans(
            n_clusters=n_groups,
            random_state=4758,
            n_init='auto',
        ).fit(
            data.T
        ).labels_

    group_grams = jax.vmap(
        lambda k: (data * (clusters == k) @ data.T)
        )(jnp.arange(n_groups)) # O(N2PK)

    cmk_data = {
        'groups': clusters,
        'data': data,
        'group_grams': group_grams,
        'n_samples': data.shape[0],
        'n_groups': n_groups,
    }
    # Convention for asymmetry:
    # D1 is the predicted, D2 the predictor
    compact_covariance0 = (
            # Ensure at least some signal everywhere
            0.1 * np.ones((n_groups, n_groups))
            # Ensure the diagonal dominates
            + 1. * jnp.eye(n_groups)
            # Make all the values a bit different to create some asymmetry
            + 0.05 * jnp.linspace(0, 1, n_groups**2).reshape((n_groups, n_groups))
        )
    # Scale to have a total signal of variance approx 0.5
    compact_covariance0 /= (
        2. * np.max(compact_covariance0 @ np.bincount(clusters, minlength=n_groups))
        )
    # About 1 but not exactly
    data_cov0 = 1.2
    cmk_init_state = {
        'compact_covariance': compact_covariance0,
        'data_vars':  np.ones(data.shape[1])*data_cov0,
    }
    return cmk_data, cmk_init_state

def cmk_factor_roots(group_grams, compact_covariance):
    """
    Compute the required factoriztions
    """
    identity_n = jnp.diag(jnp.ones_like(group_grams[0, 0]))
    root_covariances = jnp.tensordot(compact_covariance, group_grams, 1) + identity_n
    root_choleskys = lax.linalg.cholesky(root_covariances, symmetrize_input=False)
    root_inv_choleskys = lax.linalg.triangular_solve(
        *jnp.broadcast_arrays(root_choleskys, identity_n),
        lower=True
        )
    root_precisions = jnp.matmul(jnp.transpose(root_inv_choleskys, (0, 2, 1)), root_inv_choleskys)
    root_log_dets = jnp.log(jnp.diagonal(root_choleskys, axis1=1, axis2=2)).sum(-1)
    return root_precisions, root_log_dets

def cmk_many(
    # Data
    group_grams, groups, data, n_samples,
    # State
    compact_covariance, data_vars,
    **_):
    """
    Computes the essential statistics to derive updates

    (Effective parameters and model sizes mostly)
    """

    arange_k =  jnp.cumsum(jnp.ones_like(compact_covariance[0])) - 1.

    # K x P
    groups_onehot = 1. * (groups[None, :] == arange_k[:, None])

    # K x N x N, K
    root_precisions, root_log_dets = cmk_factor_roots(group_grams, compact_covariance)
    # K x K (D1: target group, D2: predictor group)
    root_eff_parameters = jnp.tensordot(
        root_precisions, group_grams.T
    ) * compact_covariance

    # K x N x P
    #all_trans_data = jnp.tensordot(root_precisions, data, 1) # !! O(K N^2 P)
    # N x P
    #trans_data = jnp.take_along_axis(all_trans_data, groups[None, None, :], 0)[0]
    trans_data = lax.map(lambda a: root_precisions[a[0]] @ a[1], (groups, data.T)).T
    # P
    own_group_covariance = jnp.take_along_axis(jnp.diagonal(compact_covariance), groups, 0)
    # P
    trans_data_data = jnp.sum(trans_data * data, 0)
    # P
    r1u_prefactor = 1. / (1. - own_group_covariance * trans_data_data)
    r1u_factors = own_group_covariance * r1u_prefactor
    # K x P
    group_covariances = jnp.take_along_axis(compact_covariance.T, groups[None, :], 1)
    # K x P
    trans_data_grams = jnp.sum(
        jnp.tensordot(group_grams,  trans_data, 1) * trans_data,
        1)  # !! O(K N^2 P)
    # K x P
    # Only the cross numbers are correct here, self-numbers needs a different calculation
    eff_parameters_cross = (
        jnp.take_along_axis(root_eff_parameters.T, groups[None, :], 1)
        +
        r1u_factors
            * group_covariances
            * trans_data_grams
    )
    # K x P
    # Correction for self
    eff_parameters = (
        eff_parameters_cross
        - groups_onehot * (
            r1u_factors * trans_data_data
        )
    )

    # P
    trans_data_ss = jnp.sum(trans_data**2, 0)
    # P
    rss = trans_data_ss * r1u_prefactor **2

    # K x P
    model_sizes_cross = group_covariances**2 * trans_data_grams * (r1u_prefactor **2)

    # P
    model_sizes = model_sizes_cross - groups_onehot * (
        trans_data_data * trans_data_data *
        r1u_factors * r1u_factors
    )


    # P
    log_likelihoods = (
        - 0.5 * n_samples * jnp.log(2. * jnp.pi)
        - 0.5 * n_samples * jnp.log(data_vars)
        - jnp.take_along_axis(root_log_dets, groups, 0)
        - 0.5 * jnp.log(1. - own_group_covariance * trans_data_data)
        - 0.5 * trans_data_data * r1u_prefactor / data_vars
    )

    # N x P (Lambda X)
    trans_targets = trans_data * r1u_prefactor /  data_vars

    return {
        'eff_params': eff_parameters,
        'rss': rss,
        'model_sizes': model_sizes,
        'log_evidence': log_likelihoods,
        'trans_target': trans_targets,
    }, {
        'group_covariances': group_covariances,
        'groups_onehot': groups_onehot,
        'own_group_covariance': own_group_covariance,
    }

cmk_many_jit = jax.jit(cmk_many)

def cmk_update(
    # Inputs
    n_samples, data_vars,
    # Intermediates
    rss, model_sizes, eff_params,
    # Aux
    group_covariances, groups_onehot,
    **_):
    """
    Derives the updates from the intermediates from `cmk_many`
    """

    tiny = jnp.finfo(model_sizes.dtype).tiny

    vanished_model_sizes = model_sizes <= tiny
    vanished_group_covariances = group_covariances <= tiny
    group_covariances = jnp.where(
        vanished_group_covariances,
        1.0,
        group_covariances
    )

    new_data_vars = (
        rss + (model_sizes / group_covariances).sum(0)
    ) / n_samples

    # K x K
    tot_eff_params =  jnp.tensordot(
            groups_onehot,
            eff_params.T,
            1)

    tiny = jnp.finfo(tot_eff_params.dtype).tiny
    vanished_eff_params = tot_eff_params <= tiny
    tot_eff_params = jnp.where(
        vanished_eff_params,
        1.0,
        tot_eff_params
    )

    tot_model_sizes =  jnp.tensordot(
            groups_onehot,
            (model_sizes / data_vars).T,
            1)
    vanished_tot_model_sizes = tot_model_sizes <= tiny

    new_compact_covariances = (
       tot_model_sizes
        / tot_eff_params
    )
    new_compact_covariances = jnp.where(
        vanished_tot_model_sizes,
        0.0,
        new_compact_covariances
    )

    return {
        'new_data_vars': new_data_vars,
        'new_compact_covariance': new_compact_covariances,
        'inf_data_vars': jnp.sum(vanished_group_covariances & (~ vanished_model_sizes)),
        'inf_compact_covariance': jnp.sum(vanished_eff_params & (~ vanished_tot_model_sizes)),
        'vanished_compact_covariance': jnp.sum(vanished_eff_params & vanished_tot_model_sizes),
    }

def cmk_predict(
    # New inputs
    new_data,
    # Inputs
    data, groups, n_groups,
    compact_covariance, data_vars,
    # Intermediates
    trans_target,
    # Aux
    own_group_covariance,
    **_):
    """
    Makes leave-one-out imputation on new data
    """

    # K x N' x N
    group_xgrams = jax.vmap(lambda k: (new_data * (groups == k) @ data.T))(jnp.arange(n_groups))

    # K x N' x N
    root_xgrams = jnp.tensordot(compact_covariance, group_xgrams, 1)

    # K x N' x P
    all_root_preds = jnp.tensordot(root_xgrams, trans_target, 1) * data_vars
    # N' x P
    root_preds = jnp.take_along_axis(all_root_preds, groups[None, None, :], 0)[0]
    # N' x P
    preds = (
        root_preds
        - jnp.sum(data * trans_target, 0) * data_vars * own_group_covariance * new_data
    )

    return preds
