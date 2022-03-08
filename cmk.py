import numpy as np
import scipy
import jax
import jax.numpy as jnp
from jax import lax

def cmk_one(
        # Target
        target, target_group,
        # Parameters
        data_var, prior_vars,
        # Pre-processed data
        group_grams
    ):
    """Target-specific MK update computations"""

    n_samples = len(target)

    group_grams_notarget = np.array(group_grams, copy=True)
    group_grams_notarget[target_group] = (
        group_grams[target_group] - target[:, None] * target[None, :]
    )
    del group_grams

    # K x N x N, weighted gram matrices
    gram_w1 = prior_vars[:, None, None] * group_grams_notarget

    # N x N
    gram_w1_sum = np.sum(gram_w1, 0)
    # N x N
    cho_data_cov, lower = scipy.linalg.cho_factor(gram_w1_sum + data_var * np.eye(n_samples))
    data_prec = scipy.linalg.cho_solve((cho_data_cov, lower), np.eye(n_samples))
    # K
    eff_params = np.sum((data_prec * gram_w1), (-2, -1))
    trans_target = data_prec @ target
    # K
    subsizes = np.sum(trans_target * (group_grams_notarget @ trans_target), -1) * prior_vars**2
    # K
    new_prior_vars = subsizes / eff_params
    # N
    preds = gram_w1_sum @ (data_prec @ target)
    # 1
    rss = np.sum((target - preds)**2)
    # 1
    new_data_var = rss / (n_samples - eff_params.sum())

    # 1
    log_evidence = (
        - 0.5 * n_samples * np.log(2*np.pi)
        - np.log(np.diagonal(cho_data_cov)).sum()
        # The two terms below sum to - N / 2 at equilibrium
        - 0.5 * rss / data_var
        - 0.5 * (subsizes / prior_vars).sum()
    )

    # trans_target is useful for predictions too
    # Note that the trans target matches the *old* data/prior_vars
    return {
        # MK updates
        'new_data_var': new_data_var,
        'new_prior_vars': new_prior_vars,
        # Objective
        'log_evidence': log_evidence,
        # Misc statistics
        'eff_params': eff_params,
        'rss': rss,
        'model_sizes': subsizes,
        # Pre-computation
        'trans_target': trans_target
    }

def cmk_one_scaled(
        # Target
        target, target_group,
        # Parameters
        data_var, prior_vars,
        # Pre-processed data
        group_grams
    ):
    """Like cmk_one, but the final prior on weights is prior_vars * data_var"""
    res = cmk_one(
        # Target
        target, target_group,
        # Parameters
        data_var, prior_vars*data_var,
        # Pre-processed data
        group_grams
    )
    del res['new_prior_vars']
    return res

def cmk_factor_roots(group_grams, compact_covariance):
    In = jnp.diag(jnp.ones_like(group_grams[0, 0]))
    root_covariances = jnp.tensordot(compact_covariance, group_grams, 1) + In
    root_choleskys = lax.linalg.cholesky(root_covariances, symmetrize_input=False)
    root_inv_choleskys = lax.linalg.triangular_solve(*jnp.broadcast_arrays(root_choleskys, In), lower=True)
    root_precisions = jnp.matmul(jnp.transpose(root_inv_choleskys, (0, 2, 1)), root_inv_choleskys)
    root_log_dets = jnp.log(jnp.diagonal(root_choleskys, axis1=1, axis2=2)).sum(-1)
    return root_precisions, root_log_dets

@jax.jit
def cmk_many(group_grams, compact_covariance, groups, values, data_vars, n_samples, n_groups):
    arange_K =  jnp.cumsum(jnp.ones_like(compact_covariance[0])) - 1.

    # K x P
    groups_onehot = 1. * (groups[None, :] == arange_K[:, None])

    # K x N x N, K
    root_precisions, root_log_dets = cmk_factor_roots(group_grams, compact_covariance)
    # K x K (D1: target group, D2: predictor group)
    root_eff_parameters = jnp.tensordot(
        root_precisions, group_grams.T
    ) * compact_covariance
    # K
    root_eff_parameters_total = jnp.sum(root_eff_parameters, 1)

    # K x N x P
    #all_trans_values = jnp.tensordot(root_precisions, values, 1) # !! O(K N^2 P)
    # N x P
    #trans_values = jnp.take_along_axis(all_trans_values, groups[None, None, :], 0)[0]
    trans_values = lax.map(lambda a: root_precisions[a[0]] @ a[1], (groups, values.T)).T
    # P
    own_group_covariance = jnp.take_along_axis(jnp.diagonal(compact_covariance), groups, 0)
    # P
    trans_values_values = jnp.sum(trans_values * values, 0)
    # P
    r1u_prefactor = 1. / (1. - own_group_covariance * trans_values_values)
    r1u_factors = own_group_covariance * r1u_prefactor
    # K x P
    group_covariances = jnp.take_along_axis(compact_covariance.T, groups[None, :], 1)
    # K x P
    trans_values_grams = jnp.sum(jnp.tensordot(group_grams,  trans_values, 1) * trans_values, 1)  # !! O(K N^2 P)
    # K x P
    # Only the cross numbers are correct here, self-numbers needs a different calculation
    eff_parameters_cross = (
        jnp.take_along_axis(root_eff_parameters.T, groups[None, :], 1)
        +
        r1u_factors
            * group_covariances
            * trans_values_grams
    )
    # K x P
    # Correction for self
    eff_parameters = (
        eff_parameters_cross
        - groups_onehot * (
            r1u_factors * trans_values_values
        )
    )

    # P
    trans_values_ss = jnp.sum(trans_values**2, 0)
    # P
    rss = trans_values_ss * r1u_prefactor **2

    # K x P
    model_sizes_cross = group_covariances**2 * trans_values_grams * (r1u_prefactor **2)

    # P
    model_sizes = model_sizes_cross - groups_onehot * (
        trans_values_values * trans_values_values *
        r1u_factors * r1u_factors
    )

    
    # P 
    log_likelihoods = (
        - 0.5 * n_samples * jnp.log(2. * jnp.pi)
        - 0.5 * n_samples * jnp.log(data_vars)
        - jnp.take_along_axis(root_log_dets, groups, 0)
        - 0.5 * jnp.log(1. - own_group_covariance * trans_values_values)
        - 0.5 * trans_values_values * r1u_prefactor / data_vars
    )

    # N x P (Lambda X)
    trans_targets = trans_values * r1u_prefactor /  data_vars

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

def cmk_update(
    # Inputs
    groups, n_samples, data_vars, compact_covariance,
    # Intermediates
    rss, model_sizes, eff_params,
    # Aux
    group_covariances, groups_onehot,
    **_):

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
    new_values,
    # Inputs
    values, groups, n_groups,
    compact_covariance, data_vars,
    # Intermediates
    trans_target,
    # Aux
    own_group_covariance,
    **_):

    # K x N' x N
    group_xgrams = jax.vmap(lambda k: (new_values * (groups == k) @ values.T))(jnp.arange(n_groups))

    # K x N' x N
    root_xgrams = jnp.tensordot(compact_covariance, group_xgrams, 1)

    # K x N' x P
    all_root_preds = jnp.tensordot(root_xgrams, trans_target, 1) * data_vars
    # N' x P
    root_preds = jnp.take_along_axis(all_root_preds, groups[None, None, :], 0)[0]
    # N' x P
    preds = (
        root_preds
        - jnp.sum(values * trans_target, 0) * data_vars * own_group_covariance * new_values
    )

    return preds
