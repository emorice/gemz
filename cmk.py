import numpy as np
import scipy

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