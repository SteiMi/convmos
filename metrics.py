import numpy as np
from scipy.stats import rv_histogram


# metrics
def correlation(preds, obs):
    # Ignore division by 0 warnings
    # I now that this can happen for static arrays but it's fine, it will simply return nan.
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.corrcoef(preds, obs)[0, 1]


def bias(preds, obs):
    return preds - obs


def mse(preds, obs):
    return np.mean(np.power(bias(preds, obs), 2))


def rmse(preds, obs):
    return np.sqrt(mse(preds, obs))


def nrmse(preds, obs):
    min_val = np.nanmin(obs)
    max_val = np.nanmax(obs)
    val_diff = max_val - min_val
    if val_diff == 0:
        return np.nan
    # According to https://link.springer.com/article/10.1007/s00704-018-2672-5
    return rmse(preds, obs) / (max_val - min_val) * 100


def combined_nrsme_rmse_mse_pbias_bias_rsq(preds, obs):
    """
    Combined calculation of several metrics. This allows us to reuse several intermediate values.
    """
    bias_ = bias(preds, obs)
    pred_sum_ = np.nansum(preds)
    if pred_sum_ == 0:
        pbias_ = np.nan
    else:
        pbias_ = 100.0 * (np.nansum(bias_) / pred_sum_)
    se_ = np.power(bias_, 2)
    mse_ = np.mean(se_)
    rmse_ = np.sqrt(mse_)

    min_val = np.nanmin(obs)
    max_val = np.nanmax(obs)

    # Don't calculate R2 and NRMSE if obs is static
    if max_val - min_val != 0.0:

        ssr_ = np.sum(se_)
        sst_ = np.sum(np.power(obs - np.mean(obs), 2))
        rsq_ = 1 - (ssr_ / sst_)

        nrmse_ = rmse_ / (max_val - min_val) * 100
    else:
        rsq_ = np.nan
        nrmse_ = np.nan

    return nrmse_, rmse_, mse_, pbias_, bias_, rsq_


def skill(preds, obs):

    y_pred_cell = preds
    y_cell = obs

    # Mask the result of this cell when there are no real values
    if len(y_pred_cell) == 0 | len(y_cell) == 0:
        return np.nan

    # span bins from min to max like described here:
    # https://rmets.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/joc.4612
    min_y = np.floor(np.minimum(np.min(y_cell), np.min(y_pred_cell)))
    max_y = np.ceil(np.maximum(np.max(y_cell), np.max(y_pred_cell)))

    # Make bins of 1 mm/day width as Perkins did
    bins = np.arange(min_y, max_y + 1, step=1, dtype=int)

    # Calculate the histogram over the time for the current cell
    y_counts_cell, bin_edges = np.histogram(y_cell, bins=bins)
    y_hist_dist_cell = rv_histogram((y_counts_cell, bin_edges))
    y_pred_hist_dist_cell = rv_histogram(np.histogram(y_pred_cell, bins=bin_edges))

    # Calculate the skill at the cell according to https://journals.ametsoc.org/doi/full/10.1175/JCLI4253.1
    skill_at_cell = 0.0
    for b in range(len(bins) - 1):
        skill_at_cell += np.minimum(
            y_pred_hist_dist_cell.pdf(bin_edges[b]), y_hist_dist_cell.pdf(bin_edges[b])
        )

    if skill_at_cell > 1.0:
        print('Skill cannot be >1. Something is wrong.')
    if skill_at_cell < 0.0:
        print('Skill cannot be <0. Something is wrong.')

    return skill_at_cell
