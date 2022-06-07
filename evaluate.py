from evaluate_rare import (
    create_bins,
    create_two_bins,
    extremely_specific_pred_paths_that_I_need_for_rare_evaluation,
)
from os.path import dirname, join, expanduser, isfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from argparse import ArgumentParser
import json
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import ImageGrid
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import xarray as xr

from metrics import combined_nrsme_rmse_mse_pbias_bias_rsq, correlation, skill
from utils import init_mpl, fisher_z_mean, set_size, flatten_dict

model_metrics_t = Dict[str, List[Dict[str, List[List[float]]]]]
temporal_model_metrics_t = Dict[str, List[Dict[str, List[float]]]]
generic_model_metrics_t = Union[model_metrics_t, temporal_model_metrics_t]


def plot_map(
    data: np.ndarray,
    vmin: float,
    vmax: float,
    title: str = '',
    graph_prefix: str = '',
    save_folder: str = '',
    log_norm: bool = False,
    no_colorbar: bool = True,
) -> None:

    aspect_ratio = 1.6545823175
    size_in_inches = 2.8
    fig, ax = plt.subplots(
        1, 1, figsize=(size_in_inches, size_in_inches * aspect_ratio)
    )  # set_size())

    norm = None
    if log_norm:
        norm = LogNorm()

    # ax.set_title(title)
    im = ax.imshow(
        data, norm=norm, extent=(-1.43, 22.22, 57.06, 42.77)
    )  # (-1.426746, 22.217735, 57.060219, 42.769917))
    im.set_clim(vmin, vmax)
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
    if not no_colorbar:
        fig.colorbar(im, ax=ax, extend='max', fraction=0.027, pad=0.04)
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter(''))
    fig.tight_layout()
    plt.savefig(join(save_folder, f'{graph_prefix}_{title}.pdf'), bbox_inches='tight')
    plt.close(fig=fig)
    # plt.show()


def calculate_metrics_flat(preds, obs) -> Dict[str, float]:
    """
    Calculate metrics for flattened targets and predictions where no spatial or time structure is present anymore.
    We use this when evaluating differently rare precipitation cases since it is not viable to compute the metrics per
    cell anymore when you only want to look at very rare events, since it is highly unlikely that multiple such events
    occured at the same cells in the test sets. This would then lead to very unstable metrics and many NaNs.
    """
    corr_ = correlation(preds, obs)
    (
        nrmse_,
        rmse_,
        mse_,
        pbias_,
        bias_,
        rsq_,
    ) = combined_nrsme_rmse_mse_pbias_bias_rsq(preds, obs)
    # We calculate the Skill Score here too but it isn't really useful as it tries to assess how well the distributions
    # fit but we are already only looking at part of the distribution.
    skill_ = skill(preds, obs)

    # Calculate a sort of accuracy for extreme precipitation
    num_samples = len(preds)
    perc_atleast_50 = np.count_nonzero(preds >= 50.0) / num_samples
    perc_under_50 = np.count_nonzero(preds < 50.0) / num_samples

    mm_ = mean_metrics(
        {
            'Correlation': corr_,
            'Bias': bias_,
            'PBias': pbias_,
            'MSE': mse_,
            'RMSE': rmse_,
            'NRMSE': nrmse_,
            'Skill score': skill_,
            '$R^2$': rsq_,
        }
    )

    mm_['Perc at least 50'] = perc_atleast_50
    mm_['Perc under 50'] = perc_under_50

    return mm_


def calculate_metrics_for_timestep(
    preds, obs
) -> Tuple[float, float, float, float, float, float, float, float]:

    # Correlation probably doesn't make too much sense per timestamp but whatever
    corr_ = correlation(preds, obs)
    (
        nrmse_,
        rmse_,
        mse_,
        pbias_,
        bias_,
        rsq_,
    ) = combined_nrsme_rmse_mse_pbias_bias_rsq(preds, obs)
    skill_ = skill(preds, obs)

    return corr_, nrmse_, rmse_, mse_, pbias_, bias_, rsq_, skill_


def calculate_metrics_for_cell(
    preds, obs
) -> Tuple[float, float, float, float, float, float, float, float]:
    # Remove data points from both REMO and E-OBS if E-OBS is nan for that day
    nan_mask = ~np.isnan(obs)
    if not any(nan_mask):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
    corr_ = correlation(preds[nan_mask], obs[nan_mask])
    (
        nrmse_,
        rmse_,
        mse_,
        pbias_,
        bias_,
        rsq_,
    ) = combined_nrsme_rmse_mse_pbias_bias_rsq(preds[nan_mask], obs[nan_mask])
    skill_ = skill(preds[nan_mask], obs[nan_mask])

    return corr_, nrmse_, rmse_, mse_, pbias_, bias_, rsq_, skill_


def calculate_metrics(
    preds, obs, processes: Optional[int] = None
) -> Dict[str, np.ndarray]:

    mask = np.load('remo_eobs_land_mask.npy')

    # for metric in [correlation, combined_nrsme_rmse_mse]:
    correlations = np.full_like(preds[0], np.nan)
    biases = np.full_like(preds[0], np.nan)
    pbiases = np.full_like(preds[0], np.nan)
    mses = np.full_like(preds[0], np.nan)
    rmses = np.full_like(preds[0], np.nan)
    nrmses = np.full_like(preds[0], np.nan)
    skills = np.full_like(preds[0], np.nan)
    rsqs = np.full_like(preds[0], np.nan)

    # First check for which cells we have to calculate metrics
    locs_to_calc = []
    args_to_calc = []
    for i in range(preds.shape[1]):
        for j in range(preds.shape[2]):

            # Only evaluate on cells where the mask is True
            # E-OBS only exists for land
            if not mask[i, j]:
                correlations[i, j] = np.nan
                biases[i, j] = np.nan
                pbiases[i, j] = np.nan
                mses[i, j] = np.nan
                rmses[i, j] = np.nan
                nrmses[i, j] = np.nan
                skills[i, j] = np.nan
                continue

            locs_to_calc.append((i, j))
            args_to_calc.append((preds[:, i, j], obs[:, i, j]))

    # Something is terribly wrong if these numbers don't match
    assert len(locs_to_calc) == len(args_to_calc)

    # Calculate the metrics for each cell in parallel
    with Pool(processes=processes) as pool:
        results = pool.starmap(calculate_metrics_for_cell, args_to_calc)

    # Map each result to the correct location
    for num, loc in enumerate(locs_to_calc):
        correlations[loc[0], loc[1]] = results[num][0]
        nrmses[loc[0], loc[1]] = results[num][1]
        rmses[loc[0], loc[1]] = results[num][2]
        mses[loc[0], loc[1]] = results[num][3]
        pbiases[loc[0], loc[1]] = results[num][4]
        biases[loc[0], loc[1]] = np.nanmean(results[num][5])
        rsqs[loc[0], loc[1]] = results[num][6]
        skills[loc[0], loc[1]] = results[num][7]

    res = {
        'Correlation': correlations.tolist(),
        'Bias': biases.tolist(),
        'PBias': pbiases.tolist(),
        'MSE': mses.tolist(),
        'RMSE': rmses.tolist(),
        'NRMSE': nrmses.tolist(),
        'Skill score': skills.tolist(),
        '$R^2$': rsqs.tolist(),
    }

    return res


def calculate_metrics_temporally(
    preds, obs, processes: Optional[int] = None
) -> Dict[str, np.ndarray]:

    mask = np.load('remo_eobs_land_mask.npy')
    # I have to invert the mask here, since I want to use numpy masked arrays and they assume True == invalid, whereas
    # I assume True == valid
    npma_mask = np.invert(mask)
    npma_mask_with_times = np.repeat(
        npma_mask[np.newaxis, :, :], preds.shape[0], axis=0
    )

    # Create masked arrays
    preds_masked = np.ma.array(preds.values, mask=npma_mask_with_times)
    obs_masked = np.ma.array(obs.values, mask=npma_mask_with_times)

    # flatten spatial dimensions
    # The compress and reshape seems to keep the right order of things according to some print-debugging
    preds_valid = preds_masked.compressed().reshape((preds.shape[0], -1))
    obs_valid = obs_masked.compressed().reshape((obs.shape[0], -1))

    args_to_calc = []
    for t in range(preds.shape[0]):
        args_to_calc.append((preds_valid[t], obs_valid[t]))

    # Calculate the metrics for each timestep in parallel
    with Pool(processes=processes) as pool:
        results = pool.starmap(calculate_metrics_for_timestep, args_to_calc)

    correlations, nrmses, rmses, mses, pbiases, biases, rsqs, skills = (
        [] for _ in range(8)
    )

    # Map each result to the correct timestep
    for t in range(preds.shape[0]):
        # Not sure why I have to cast explicitly here for json.dump to work
        correlations.append(float(results[t][0]))
        nrmses.append(float(results[t][1]))
        rmses.append(float(results[t][2]))
        mses.append(float(results[t][3]))
        pbiases.append(float(results[t][4]))
        biases.append(float(np.nanmean(results[t][5])))
        rsqs.append(float(results[t][6]))
        skills.append(float(results[t][7]))

    res = {
        'Correlation': correlations,
        'Bias': biases,
        'PBias': pbiases,
        'MSE': mses,
        'RMSE': rmses,
        'NRMSE': nrmses,
        'Skill score': skills,
        '$R^2$': rsqs,
    }

    return res


def mean_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:

    return {
        'Correlation': fisher_z_mean(
            np.array(metrics['Correlation'])[
                ~np.isnan(np.array(metrics['Correlation']))
            ]
        ),
        'Bias': np.nanmean(metrics['Bias']),
        'PBias': np.nanmean(metrics['PBias']),
        'MSE': np.nanmean(metrics['MSE']),
        'RMSE': np.nanmean(metrics['RMSE']),
        'NRMSE': np.nanmean(metrics['NRMSE']),
        'Skill score': np.nanmean(metrics['Skill score']),
        '$R^2$': np.nanmean(metrics['$R^2$']),
    }


def eval_and_plot(
    predictions: xr.Dataset,
    graph_save_folder: str,
    season_name: str = 'all',
    log_norm: bool = False,
    processes: Optional[int] = None,
):

    print(f'Model results (Season {season_name}):')
    model_metrics = calculate_metrics(
        predictions.pred, predictions.target, processes=processes
    )
    print(mean_metrics(model_metrics))

    print(f'Baseline results (Season {season_name}):')
    baseline_metrics = calculate_metrics(
        predictions.input, predictions.target, processes=processes
    )
    print(mean_metrics(baseline_metrics))

    for metric_name, metric_values in baseline_metrics.items():

        base_min = np.nanmin(metric_values)
        base_max = np.nanmax(metric_values)
        model_min = np.nanmin(model_metrics[metric_name])
        model_max = np.nanmax(model_metrics[metric_name])

        # REMO sometimes has outliers that destroy the colormap
        # I try to fix that by using "only" the 95th percentile as base_max in such occasions
        base_range = base_max - base_min
        base_95perc = np.nanpercentile(metric_values, 95)
        outlier_perc = (base_95perc - base_min) / base_range
        print(
            metric_name,
            'base_range:',
            base_range,
            'base_95perc:',
            base_95perc,
            'outlier_perc:',
            outlier_perc,
        )
        if outlier_perc <= 0.9:
            print('Outliers detected!')
            base_max = base_95perc

        vmin = base_min if base_min < model_min else model_min
        vmax = base_max if base_max > model_max else model_max

        if metric_name == 'RMSE':
            # vmax = np.nanmax(metric_values)
            vmax = 10.0

        if metric_name == '$R^2$' and vmin < -0.5:
            vmin = -0.5

        cur_log_norm = log_norm
        if log_norm and (vmin < 0.0 or vmax < 0.0):
            print(
                'Ignoring log_norm since there are negative values for metric',
                metric_name,
            )
            cur_log_norm = False

        np.savetxt(
            join(graph_save_folder, f'model_{season_name}_{metric_name}.csv'),
            model_metrics[metric_name],
            delimiter=',',
        )

        plot_map(
            metric_values,
            vmin=vmin,
            vmax=vmax,
            title=metric_name,
            graph_prefix=f'baseline_{season_name}',
            save_folder=graph_save_folder,
            log_norm=cur_log_norm,
            no_colorbar=False,
        )
        plot_map(
            model_metrics[metric_name],
            vmin=vmin,
            vmax=vmax,
            title=metric_name,
            graph_prefix=f'model_{season_name}',
            save_folder=graph_save_folder,
            log_norm=cur_log_norm,
            no_colorbar=False,
        )
        plot_map(
            metric_values,
            vmin=vmin,
            vmax=vmax,
            title=metric_name,
            graph_prefix=f'baseline_nocb_{season_name}',
            save_folder=graph_save_folder,
            log_norm=cur_log_norm,
            no_colorbar=True,
        )
        plot_map(
            model_metrics[metric_name],
            vmin=vmin,
            vmax=vmax,
            title=metric_name,
            graph_prefix=f'model_nocb_{season_name}',
            save_folder=graph_save_folder,
            log_norm=cur_log_norm,
            no_colorbar=True,
        )


def evaluate(
    prediction_file: str,
    seasonal: bool = False,
    log_norm: bool = False,
    processes: Optional[int] = None,
):

    predictions = xr.open_dataset(prediction_file).load()
    print(predictions)

    graph_save_folder = dirname(prediction_file)

    if seasonal:
        for season in ['DJF', 'JJA', 'MAM', 'SON']:
            # preds_in_season = predictions[predictions['time.season'] == season]
            preds_in_season = predictions.sel(time=predictions['time.season'] == season)
            eval_and_plot(
                preds_in_season,
                graph_save_folder=graph_save_folder,
                season_name=season,
                log_norm=log_norm,
                processes=processes,
            )

    else:
        eval_and_plot(
            predictions,
            graph_save_folder=graph_save_folder,
            log_norm=log_norm,
            processes=processes,
        )


def plot_grid(
    model_metrics: model_metrics_t,
    baseline_metrics: Dict[str, str],
    metric: str,
    vmin: float,
    vmax: float,
    output_suffix: str = '',
    output_prefix: str = '',
):
    # I use 0.95*\textwidth
    width_in_inches = 7.0056039851
    fig = plt.figure(
        figsize=(width_in_inches, 100)
    )  # Height simply has to be large enough due to aspect in imshow

    grid = ImageGrid(
        fig,
        111,
        # nrows_ncols=(2, 3),
        # nrows_ncols=(5, 3),
        nrows_ncols=(4, 3),
        axes_pad=0.35,
        share_all=True,
        cbar_location='right',
        cbar_mode='single',
        cbar_size='3%',
        # cbar_size='5%',
        cbar_pad=0.15,
    )

    norm = None

    axs = list(grid)

    im = axs[0].imshow(
        # np.random.random((10, 10)),
        baseline_metrics[metric],
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extent=(-1.43, 22.22, 57.06, 42.77),
        aspect=1.6545823175,
    )
    axs[0].set_title('REMO raw')
    axs[0].invert_yaxis()
    axs[0].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[0].xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
    axs[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))

    # Write a dummy image to get the spacing right
    axs[2].imshow(
        # np.random.random((10, 10)),
        baseline_metrics[metric],
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extent=(-1.43, 22.22, 57.06, 42.77),
        aspect=1.6545823175,
    )

    # The model's are plotted in the order they are defined in paper_pred_paths in the paper_figure function.
    for i, model in enumerate(model_metrics.keys()):
        if model != 'ConvMOS':
            # This forces the baselines to be on the second row, given the current order of the plots.
            i += 1
        ax = axs[i + 1]
        ax.imshow(
            # np.random.random((10, 10)),
            model_metrics[model][0][metric],
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            extent=(-1.43, 22.22, 57.06, 42.77),
            aspect=1.6545823175,
        )
        # ax.imshow(model_metrics[model][0][metric], vmin=vmin, vmax=vmax, norm=norm,
        #           extent=(-1.43, 22.22, 57.06, 42.77), aspect=1.6545823175)
        ax.set_title(model)
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    # We don't need the last one in the first row
    axs[2].set_visible(False)
    cbar = axs[-2].cax.colorbar(im, extend='max')
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor('face')
    fig.tight_layout()
    plt.savefig(
        join('eval_out', f'{output_prefix}{metric}_maps{output_suffix}.pdf'),
        bbox_inches='tight',
        dpi=200,
    )
    plt.close(fig=fig)


def plot_histogram(model_preds: Dict[str, xr.Dataset], output_prefix: str = ''):
    # Create Histogram only for locations with E-OBS data (i.e. land locations)
    mask = np.load('remo_eobs_land_mask.npy')
    labels = ['E-OBS', 'REMO']
    # .values.ravel() converts the data to a 1D array which speeds things up extremely
    data = [
        list(model_preds.values())[0].target.values[:, mask].ravel(),
        list(model_preds.values())[0].input.values[:, mask].ravel(),
    ]
    for model, model_data in model_preds.items():
        labels.append(model)
        data.append(model_data.pred.values[:, mask].ravel())

    # adjusted tab20 color cycle
    tab20_data = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # 1f77b4
        (1.0, 0.4980392156862745, 0.054901960784313725),  # ff7f0e
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # 2ca02c
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # d62728
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # 9467bd
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # 8c564b
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # e377c2
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # 7f7f7f
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # bcbd22
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # 17becf
        (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),  # aec7e8
        # (1.0, 0.7333333333333333, 0.47058823529411764),  # ffbb78
        (0.8828125, 0.707023125, 0.46484375),  # e2b577 (Replacement for ffbb78)
        (0.596078431372549, 0.8745098039215686, 0.5411764705882353),  # 98df8a
        (1.0, 0.596078431372549, 0.5882352941176471),  # ff9896
        (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),  # c5b0d5
        (0.7686274509803922, 0.611764705882353, 0.5803921568627451),  # c49c94
        (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),  # f7b6d2
        (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),  # c7c7c7
        (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),  # dbdb8d
        (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),  # 9edae5
    ]

    # fig, ax = plt.subplots(figsize=set_size(fraction=1))
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=set_size(fraction=1),
        gridspec_kw={'width_ratios': [2, 1]},
    )
    ax.set_xticks(np.arange(0, 600, 50))
    # ax.margins(x=0.01)
    # plt.grid(axis='y', linewidth=0.5)
    ax.hist(
        data,
        label=labels,
        histtype='bar',
        bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
        log=True,
        color=tab20_data[: len(labels)],
    )
    ax2.hist(
        data[
            :3
        ],  # Using only 3 models makes the bar wider and there is no other data there anyways
        label=labels[:3],
        histtype='bar',
        bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
        log=True,
        color=tab20_data[:3],
    )
    ax.set_xlim((0, 200))
    ax2.set_xlim((200, 600))
    ax.legend(ncol=3, loc='upper right', bbox_to_anchor=(1.5, 1.0), fontsize='x-small')
    # ax.set_xlabel('Precipitation [mm]')
    fig.text(0.5, -0.01, 'Precipitation [mm]', ha='center')
    # fig.tight_layout()
    ax.grid(axis='y', linewidth=0.5)
    ax2.grid(axis='y', linewidth=0.5)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.get_yaxis().tick_left()
    # ax.tick_params(labelright='off')
    ax2.get_yaxis().tick_right()
    # ax2.yaxis.set_label_position('right')
    ax2.get_yaxis().set_label_position('right')
    # ax2.yaxis.set_ticklabels([])
    # ax2.yaxis.set_visible(False)

    d = 0.015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    # ax2.plot((-d,+d), (-d,+d), **kwargs)

    ax2.set_xticks(ax2.get_xticks()[1:])
    ax.set_zorder(10)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.0)

    plt.savefig(
        join('eval_out', output_prefix + 'histogram.pdf'), bbox_inches='tight', dpi=200
    )
    plt.close(fig=fig)


def plot_temporally(
    stacked_temporal_model_metrics: Dict[str, Dict[str, List[List[float]]]],
    temporal_baseline_metrics: Dict[str, List[float]],
    metric: str = 'RMSE',
    output_prefix: str = '',
    smooth_windows: int = 14,
    start_date: str = '2011-01-01',
    end_date: str = '2015-12-31',
    ignore_models: List[str] = [
        'ResNet18',
        'ResNet34',
        'ResNet50',
        'ResNet101',
        'Lin',
        'NL RF',
        'CM U-Net',
        'U-Net',
    ],
):

    fig, ax = plt.subplots(
        1,
        1,
        figsize=set_size(fraction=1),
    )
    ax.margins(x=0)

    x = pd.date_range(start=start_date, end=end_date, freq='D')

    def movingaverage(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')

    baseline_y = temporal_baseline_metrics[metric]
    if smooth_windows > 0:
        baseline_y = movingaverage(baseline_y, smooth_windows)

    ax.plot(x, baseline_y, label='REMO raw')

    for model, metrics in stacked_temporal_model_metrics.items():
        if model in ignore_models:
            continue
        np_metrics = np.array(metrics[metric])
        y = np_metrics.mean(axis=0)
        if smooth_windows > 0:
            y = movingaverage(y, smooth_windows)
        ax.plot(x, y, label=model)

    fig.tight_layout()
    if metric == 'RMSE':
        ax.set_ylabel('RMSE [mm]')
    else:
        ax.set_ylabel(metric)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

    plt.legend()
    plt.savefig(
        join('eval_out', output_prefix + 'temporal_' + metric + '.pdf'),
        bbox_inches='tight',
        dpi=200,
    )
    plt.close(fig=fig)


def ablation_pred_paths(
    path_prefix: str = join(
        '/', 'scratch', 'steininger', 'deepsd', 'scratch_remo', 'ablation'
    ),
) -> Dict[str, List[str]]:
    pred_paths = defaultdict(list)
    print(path_prefix)

    for path in Path(path_prefix).rglob('predictions_test_*.nc'):
        path_str = str(path)

        # Assume that the modules are specified between the last and second-to-last "-"
        last_minus_ind = path_str.rfind('-')
        secondlast_minus_ind = path_str.rfind('-', 0, last_minus_ind)
        modules = path_str[secondlast_minus_ind + 1 : last_minus_ind]

        if modules == '':
            print('WARNING: Empty modules for path ' + path_str)

        pred_paths[modules].append(path_str)

    return dict(pred_paths)


def main_pred_paths(
    path_prefix: str = join(
        '/', 'scratch', 'steininger', 'deepsd', 'convmos_paper_runs'
    ),
    alpha: str = '0.0',
) -> Dict[str, List[str]]:
    # The first .nc file of each model's list bears special significance as this one is used for plotting maps.
    # I always use the best model instance for each model as the first one to show best possible results.
    pred_paths = {
        'ConvMOS': [
            expanduser(join(path_prefix, alpha, 'convmos_gggl_13_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_1_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_2_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_3_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_4_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_5_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_6_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_7_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_8_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_9_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_10_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_11_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_12_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_14_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_15_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_16_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_17_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_18_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_19_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_gggl_20_test_predictions.nc')),
        ],
        'Lin': [expanduser(join(path_prefix, alpha, 'linear_test_predictions.nc'))],
        'NL PCR': [expanduser(join(path_prefix, alpha, 'nlpcr_test_predictions.nc'))],
        'NL RF': [
            expanduser(join(path_prefix, alpha, 'nlrf_19_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_1_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_2_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_3_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_4_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_5_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_6_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_7_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_8_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_9_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_10_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_11_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_12_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_13_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_14_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_15_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_16_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_17_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_18_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'nlrf_20_test_predictions.nc')),
        ],
        'U-Net': [
            expanduser(join(path_prefix, alpha, 'unet_15_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_1_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_2_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_3_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_4_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_5_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_6_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_7_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_8_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_9_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_10_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_11_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_12_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_13_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_14_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_16_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_17_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_18_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_19_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'unet_20_test_predictions.nc')),
        ],
        'CM U-Net': [
            expanduser(join(path_prefix, alpha, 'convmos_unet_11_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_1_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_2_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_3_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_4_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_5_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_6_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_7_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_8_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_9_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_10_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_12_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_13_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_14_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_15_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_16_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_17_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_18_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_19_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'convmos_unet_20_test_predictions.nc')),
        ],
        'ResNet18': [
            expanduser(join(path_prefix, alpha, 'resnet18_20_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_1_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_2_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_3_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_4_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_5_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_6_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_7_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_8_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_9_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_10_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_11_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_12_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_13_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_14_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_15_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_16_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_17_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_18_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet18_19_test_predictions.nc')),
        ],
        'ResNet34': [
            expanduser(join(path_prefix, alpha, 'resnet34_15_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_1_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_2_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_3_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_4_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_5_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_6_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_7_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_8_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_9_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_10_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_11_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_12_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_13_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_14_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_16_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_17_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_18_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_19_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet34_20_test_predictions.nc')),
        ],
        'ResNet50': [
            expanduser(join(path_prefix, alpha, 'resnet50_1_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_2_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_3_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_4_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_5_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_6_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_7_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_8_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_9_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_10_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_11_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_12_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_13_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_14_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_15_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_16_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_17_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_18_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_19_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet50_20_test_predictions.nc')),
        ],
        'ResNet101': [
            expanduser(join(path_prefix, alpha, 'resnet101_18_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_1_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_2_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_3_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_4_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_5_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_6_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_7_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_8_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_9_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_10_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_11_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_12_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_13_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_14_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_15_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_16_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_17_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_19_test_predictions.nc')),
            expanduser(join(path_prefix, alpha, 'resnet101_20_test_predictions.nc')),
        ],
    }
    return pred_paths


def model_metrics_to_mean_model_metrics(
    model_metrics: model_metrics_t,
) -> Dict[Any, Dict[str, List[float]]]:
    tmp_mean_model_metrics = {
        k: [mean_metrics(v) for v in v_l] for k, v_l in model_metrics.items()
    }
    mean_model_metrics: Dict[str, Dict[str, List[float]]] = {}
    for m, v in tmp_mean_model_metrics.items():
        # Merge list of dicts to dict of lists (https://stackoverflow.com/a/52693367/9083711)
        mmm_list = defaultdict(list)
        for d in v:
            for k, mmv in d.items():
                mmm_list[k].append(mmv)
        mean_model_metrics[m] = mmm_list

    return mean_model_metrics


def stack_temporal_model_metrics(
    temporal_model_metrics: temporal_model_metrics_t,
) -> Dict[Any, Dict[str, List[List[float]]]]:
    stacked_model_metrics: Dict[str, Dict[str, List[List[float]]]] = {}

    for m, v in temporal_model_metrics.items():
        # Merge list of dicts to dict of lists (https://stackoverflow.com/a/52693367/9083711)
        mm_list = defaultdict(list)
        for d in v:
            for k, mmv in d.items():
                mm_list[k].append(mmv)
        stacked_model_metrics[m] = mm_list

    return stacked_model_metrics


def get_model_metrics(
    paper_preds: Dict[str, List[xr.Dataset]],
    processes: Optional[int] = None,
    output_prefix: str = '',
) -> model_metrics_t:
    """Calculate the model metrics per cell and cache them. If the model_metrics were already calculated at some point
    then simply load the cached file."""
    # cached_mm_file = join('eval_out', output_prefix + 'model_metrics.pkl')
    cached_mm_file = join('eval_out', output_prefix + 'model_metrics.json')

    if isfile(cached_mm_file):
        print(f'Loading cached model metrics from {cached_mm_file}')
        # model_metrics = pickle.load(open(cached_mm_file, 'rb'))
        model_metrics = json.load(open(cached_mm_file, 'r'))
    else:
        model_metrics = {
            k: [calculate_metrics(v.pred, v.target, processes=processes) for v in v_l]
            for k, v_l in paper_preds.items()
        }
        # pickle.dump(model_metrics, open(cached_mm_file, 'wb'))
        json.dump(model_metrics, open(cached_mm_file, 'w'))

    return model_metrics


def get_temporal_model_metrics(
    paper_preds: Dict[str, List[xr.Dataset]],
    processes: Optional[int] = None,
    output_prefix: str = '',
) -> temporal_model_metrics_t:
    """Calculate the model metrics over time and cache them. If the model_metrics were already calculated at some point
    then simply load the cached file."""
    cached_mm_file = join('eval_out', output_prefix + 'temporal_model_metrics.json')

    if isfile(cached_mm_file):
        print(f'Loading cached model metrics from {cached_mm_file}')
        temporal_model_metrics = json.load(open(cached_mm_file, 'r'))
    else:
        temporal_model_metrics = {
            k: [
                calculate_metrics_temporally(v.pred, v.target, processes=processes)
                for v in v_l
            ]
            for k, v_l in paper_preds.items()
        }
        json.dump(temporal_model_metrics, open(cached_mm_file, 'w'))

    return temporal_model_metrics


def condense_model_metrics(
    mean_model_metrics: Dict[Any, Dict[str, List[float]]],
    metric: str,
    num_runs: int = 20,
    quiet: bool = False,
) -> Dict[Any, List[float]]:
    condensed_model_metrics = {}
    print('metric:', metric)
    for model, metrics in mean_model_metrics.items():
        mm = metrics[metric]
        # model can be a simple string or a tuple with (alpha, modelname)
        if (
            model == 'Lin'
            or model == 'NL PCR'
            or (
                isinstance(model, tuple) and (model[1] == 'Lin' or model[1] == 'NL PCR')
            )
        ):
            # Lin and NL PCR are deterministic and will always yield the same results.
            # Instead of calculating 20 times the same metrics, I will just calculate them once and append them
            # 20 times
            condensed_model_metrics[model] = mm * num_runs
        else:
            condensed_model_metrics[model] = mm
        if not quiet:
            if metric == 'Correlation':
                print(model, '\t', fisher_z_mean(np.array(mm)), f'({np.nanstd(mm)})')
            else:
                print(model, '\t', np.nanmean(mm), f'({np.nanstd(mm)})')
    return condensed_model_metrics


def multi_rare_evaluation(
    pred_paths_per_alpha: Dict[str, Dict[str, List[str]]],
    processes: Optional[int] = None,
    histogram: bool = True,
    two_bin_eval: bool = False,
    # output_prefix: str = '',
):
    print('Loading prediction files...')
    paper_preds_per_alpha: Dict[str, Dict[str, List[xr.Dataset]]] = {}
    for alpha, models in pred_paths_per_alpha.items():
        for k, paths in models.items():
            paper_preds_per_alpha[alpha] = {
                k: [xr.open_dataset(path).load() for path in paths]
            }

    if histogram:
        print('Plot distributions...')
        to_plot: Dict[str, xr.Dataset] = {
            'ConvMOS': paper_preds_per_alpha['0.0']['ConvMOS'][0],
            'ConvMOS-DL': paper_preds_per_alpha['1.0']['ConvMOS'][0],
        }
        plot_histogram(to_plot)

    print('Flatten data for rare evalution...')
    # If we don't flatten for rare evaluation and calculate metrics per cell instead, we have way too few rare samples
    # at many locations. This leads to absurdely unstable metrics and a lot of NaNs.
    some_alpha = next(iter(paper_preds_per_alpha.keys()))
    some_model_name = next(iter(paper_preds_per_alpha[some_alpha].keys()))
    some_dataset = paper_preds_per_alpha[some_alpha][some_model_name][0]
    mask = np.load('remo_eobs_land_mask.npy')
    # Get and flatten targets once since they are the same for all models
    targets = some_dataset.target.values[:, mask].ravel()
    paper_flatpreds_per_alpha: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for alpha, models in paper_preds_per_alpha.items():  # type: ignore
        for name, xr_datasets in models.items():
            paper_flatpreds_per_alpha[alpha] = {
                name: [xr_dataset.pred.values[:, mask].ravel() for xr_dataset in xr_datasets]  # type: ignore
            }
    # Add REMO as a baseline
    baseline_flatpreds = some_dataset.input.values[:, mask].ravel()
    paper_flatpreds_per_alpha['None'] = {'Baseline': [baseline_flatpreds]}
    # print('Any targets NaN?', np.isnan(targets).any(), some_dataset.target.values.shape, targets.shape, np.count_nonzero(mask == False))
    # some_flatpreds = next(iter(paper_flatpreds_per_alpha[some_alpha].values()))[0]
    # print('Any preds NaN?', np.isnan(some_flatpreds).any(), some_dataset.pred.values.shape, some_flatpreds.shape, np.count_nonzero(mask == False))

    # Free up some memory
    del paper_preds_per_alpha

    print('Filter by rarity...')
    # Here we could either bin and use bin ranks or use percentiles (the latter is probably easier to understand)
    # mask = np.load('remo_eobs_land_mask.npy')
    # _, bin_infos = create_bins(some_dataset.target, 20)
    if two_bin_eval:
        print('Evaluating on two bins')
        _, bin_infos = create_two_bins(
            targets, 50
        )  # DWD calls daily precipitation >= 50mm "Ergiebiger Dauerregen"
    else:
        print('Evaluating on five equidistant bins')
        _, bin_infos = create_bins(targets, 20)
    # print(test_bins)
    print(bin_infos)

    for i, row in bin_infos.iterrows():
        print()
        print(
            f'Bin {row["bin_low"]} - {row["bin_high"]} (y: {row["y_low"]} - {row["y_high"]}), size: {row["count"]}'
        )

        # Assume that the bin mask is the same for all runs of all models since we bin according to the targets and
        # the targets are the same
        bin_mask = (targets >= row['y_low']) & (targets < row['y_high'])

        # Filter targets for this bin
        targets_bin = targets[bin_mask]

        # Basically copy "paper_preds_per_alpha" but now with nan's everywhere this is not within the bin to consider
        paper_flatpreds_per_alpha_bin: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for alpha, model_preds_nd in paper_flatpreds_per_alpha.items():
            print(f'Filtering data ({alpha})...')
            paper_flatpreds_per_alpha_bin[alpha] = {}
            for model_name, preds in model_preds_nd.items():
                paper_flatpreds_per_alpha_bin[alpha][model_name] = [
                    p[bin_mask] for p in preds
                ]

        print('Calculating metrics...')

        # alpha -> Model -> Metric -> [20 values of that metric (one per run)
        mean_model_metrics_per_alpha: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for alpha, model_preds_nd in paper_flatpreds_per_alpha_bin.items():
            mean_model_metrics_per_alpha[alpha] = {}
            for model_name, preds in model_preds_nd.items():

                # Calculate the metrics per model run in parallel
                args_to_calc = [(pred, targets_bin) for pred in preds]
                with Pool(processes=processes) as pool:
                    list_of_metrics = pool.starmap(calculate_metrics_flat, args_to_calc)
                # list_of_metrics = [
                #     calculate_metrics_flat(pred, targets_bin)
                #     for pred in preds
                # ]
                # Convert List of Dicts to Dict of Lists (see https://stackoverflow.com/a/33046935)
                mean_model_metrics_per_alpha[alpha][model_name] = {
                    k: [dic[k] for dic in list_of_metrics] for k in list_of_metrics[0]
                }

        # Flatten mean_model_metrics_per_alpha dict so that we have
        # (alpha, model) -> metric -> [20 values of that metric (one per run)]
        mean_model_metrics = flatten_dict(mean_model_metrics_per_alpha)
        print(mean_model_metrics)
        if i == 0 and two_bin_eval:
            non_extreme_mean_model_metrics: Dict[
                Tuple[str, str], Dict[str, List[float]]
            ] = mean_model_metrics
        elif i == 1 and two_bin_eval:
            extreme_mean_model_metrics: Dict[
                Tuple[str, str], Dict[str, List[float]]
            ] = mean_model_metrics

        # alphas = list(mean_model_metrics_per_alpha.keys())

        for metric in [
            'Correlation',
            'Bias',
            'RMSE',
            'Skill score',
            '$R^2$',
            'Perc at least 50',
            'Perc under 50',
        ]:

            condensed_model_metrics = condense_model_metrics(
                mean_model_metrics,
                metric=metric,
                num_runs=len(paper_flatpreds_per_alpha[some_alpha][some_model_name]),
            )

            # Significance test
            if ('1.0', 'ConvMOS') in condensed_model_metrics:
                for model in condensed_model_metrics.keys():
                    if model != ('1.0', 'ConvMOS'):
                        try:
                            _, wilc_p = wilcoxon(
                                condensed_model_metrics[('1.0', 'ConvMOS')],
                                condensed_model_metrics[model],
                            )
                        except ValueError:
                            print(
                                f'Could not calculate Wilcoxon Test for {metric} between {model} and (1.0, ConvMOS)'
                            )
                            wilc_p = np.nan

                        if wilc_p > 0.05:
                            wilc_significant = False
                        else:
                            wilc_significant = True
                        print(
                            metric,
                            '(1.0, ConvMOS) vs.',
                            model,
                            ', Wilcoxon p:',
                            wilc_p,
                            'significant?',
                            wilc_significant,
                        )

    # Calculate balanced Accuracy (positive = extreme, negative = non-extreme)
    print()
    print('Balanced Accuracy...')
    tprs = condense_model_metrics(
        extreme_mean_model_metrics,
        metric='Perc at least 50',
        num_runs=len(paper_flatpreds_per_alpha[some_alpha][some_model_name]),
        quiet=True,
    )
    tnrs = condense_model_metrics(
        non_extreme_mean_model_metrics,
        metric='Perc under 50',
        num_runs=len(paper_flatpreds_per_alpha[some_alpha][some_model_name]),
        quiet=True,
    )
    balanced_accuracies = {}
    for model in tprs.keys():
        balanced_accuracies[model] = (
            np.array(tprs[model]) + np.array(tnrs[model])
        ) / 2.0
        print(
            model,
            'Balanced Accuracy:',
            np.mean(balanced_accuracies[model]),
            f'({np.std(balanced_accuracies[model])})',
        )
    for model in tprs.keys():
        if model != ('1.0', 'ConvMOS'):
            try:
                _, wilc_p = wilcoxon(
                    condensed_model_metrics[('1.0', 'ConvMOS')],
                    condensed_model_metrics[model],
                )
            except ValueError:
                print(
                    f'Could not calculate Wilcoxon Test for Balanced Accuracy between {model} and (1.0, ConvMOS)'
                )
                wilc_p = np.nan

            if wilc_p > 0.05:
                wilc_significant = False
            else:
                wilc_significant = True
            print(
                'Balanced Accuracy',
                '(1.0, ConvMOS) vs.',
                model,
                ', Wilcoxon p:',
                wilc_p,
                'significant?',
                wilc_significant,
            )


def multi_evaluation(
    pred_paths: Dict[str, List[str]],
    processes: Optional[int] = None,
    histogram: bool = True,
    output_prefix: str = '',
):

    print('Loading prediction files...')
    paper_preds = {
        k: [xr.open_dataset(path).load() for path in paths]  # take all files
        # k: [xr.open_dataset(paths[0])]  # take only first file
        for k, paths in pred_paths.items()
    }

    if histogram:
        print('Plot distributions...')
        plot_histogram({k: v_l[0] for k, v_l in paper_preds.items()})

    print('Calculating metrics...')
    temporal_model_metrics = get_temporal_model_metrics(
        paper_preds, processes=processes, output_prefix=output_prefix
    )
    stacked_temporal_model_metrics = stack_temporal_model_metrics(
        temporal_model_metrics
    )

    # Just use the first best dataset for the baseline data, as it is in every .nc file anyway
    some_dataset = next(iter(paper_preds.values()))[0]
    temporal_baseline_metrics = calculate_metrics_temporally(
        some_dataset.input, some_dataset.target
    )
    baseline_metrics = calculate_metrics(some_dataset.input, some_dataset.target)

    model_metrics = get_model_metrics(
        paper_preds, processes=processes, output_prefix=output_prefix
    )

    mean_model_metrics = model_metrics_to_mean_model_metrics(model_metrics)

    num_runs = max([len(preds) for preds in paper_preds.values()])

    # Infer start and end date for temporal plot
    start_date = some_dataset.time.min()
    end_date = some_dataset.time.max()
    start_date_str = str(start_date.dt.strftime('%Y-%m-%d').values)
    end_date_str = str(end_date.dt.strftime('%Y-%m-%d').values)

    for metric in ['Correlation', 'Bias', 'RMSE', 'Skill score', '$R^2$', 'NRMSE']:

        # Create the temporal plots
        plot_temporally(
            stacked_temporal_model_metrics,
            temporal_baseline_metrics,
            metric=metric,
            start_date=start_date_str,
            end_date=end_date_str,
            output_prefix=output_prefix,
        )

        condensed_model_metrics = condense_model_metrics(
            mean_model_metrics,
            metric=metric,
            num_runs=num_runs,  # len(paper_preds['ConvMOS'])
        )

        # Significance test
        if 'ConvMOS' in model_metrics:
            for model in model_metrics.keys():
                if model != 'ConvMOS':
                    _, wilc_p = wilcoxon(
                        condensed_model_metrics['ConvMOS'],
                        condensed_model_metrics[model],
                    )
                    if wilc_p > 0.05:
                        wilc_significant = False
                    else:
                        wilc_significant = True
                    print(
                        metric,
                        'ConvMOS vs.',
                        model,
                        ', Wilcoxon p:',
                        wilc_p,
                        'significant?',
                        wilc_significant,
                    )

        if 'CM U-Net' in model_metrics:
            for model in model_metrics.keys():
                if model != 'CM U-Net':
                    _, wilc_p = wilcoxon(
                        condensed_model_metrics['CM U-Net'],
                        condensed_model_metrics[model],
                    )
                    if wilc_p > 0.05:
                        wilc_significant = False
                    else:
                        wilc_significant = True
                    print(
                        metric,
                        'CM U-Net vs.',
                        model,
                        ', Wilcoxon p:',
                        wilc_p,
                        'significant?',
                        wilc_significant,
                    )

    vmin = 0.0
    vmax = 10.0
    # model_metrics = {'ConvMOS': [], 'Lin': [], 'NL PCR': [], 'NL RF': []}
    # baseline_metrics = {}

    metric = 'RMSE'
    plot_grid(
        model_metrics,
        baseline_metrics,
        metric,
        vmin,
        vmax,
        output_prefix=output_prefix,
        output_suffix='_morebaselines',
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        'prediction_file',
        type=str,
        nargs='?',
        help='Path to a predictions.nc file (generated from run.py)',
    )
    parser.add_argument(
        '-l', '--log-norm', action='store_true', help='Log-normalize the images'
    )
    parser.add_argument(
        '-a', '--ablation', action='store_true', help='Evaluate ablation study'
    )
    parser.add_argument(
        '-r',
        '--rare',
        action='store_true',
        help='Evaluate performance for rare data points with DenseLoss',
    )
    parser.add_argument(
        '-t',
        '--two-bin',
        action='store_true',
        help='Evaluate performance for rare data points for two bins instead of 5 equidistant bins (only with -r)',
    )
    parser.add_argument(
        '-p',
        '--processes',
        type=int,
        default=8,
        help='Number of processes (default: 8)',
    )
    parser.add_argument(
        '--path-prefix',
        type=str,
        default=join('~', 'convmos_rmse_maps'),
        help='Path prefix to the .nc files for the paper figures',
    )
    args = parser.parse_args()

    init_mpl()

    if args.prediction_file is None:

        if args.ablation:
            print('Evaluating ablation study..')
            pred_paths = ablation_pred_paths(path_prefix=args.path_prefix)
            output_prefix = 'ablation_'
            histogram = False

            multi_evaluation(
                pred_paths,
                processes=args.processes,
                output_prefix=output_prefix,
                histogram=histogram,
            )

        elif args.rare:
            print('Evaluating performance for rare data points with DenseLoss..')
            pred_paths_per_alpha: Dict[
                str, Dict[str, List[str]]
            ] = extremely_specific_pred_paths_that_I_need_for_rare_evaluation()

            # alphas = ['0.0', '1.0']
            # pred_paths_per_alpha: Dict[str, Dict[str, List[str]]] = {}
            # for alpha in alphas:
            #     pred_paths_per_alpha[alpha] = main_pred_paths(
            #         path_prefix=args.path_prefix, alpha=alpha
            #     )

            # For Testing:
            # pred_paths_per_alpha = {
            #     '0.0': {
            #         'ConvMOS': [
            #             expanduser(
            #                 join(
            #                     '~',
            #                     'convmos_rmse_maps',
            #                     '0.0',
            #                     'convmos_gggl_1_test_predictions.nc',
            #                 )
            #             ),
            #             expanduser(
            #                 join(
            #                     '~',
            #                     'convmos_rmse_maps',
            #                     '0.0',
            #                     'convmos_gggl_2_test_predictions.nc',
            #                 )
            #             ),
            #         ]
            #     },
            #     '1.0': {
            #         'ConvMOS': [
            #             expanduser(
            #                 join(
            #                     '~',
            #                     'convmos_rmse_maps',
            #                     '1.0',
            #                     'convmos_gggl_1_test_predictions.nc',
            #                 )
            #             ),
            #             expanduser(
            #                 join(
            #                     '~',
            #                     'convmos_rmse_maps',
            #                     '1.0',
            #                     'convmos_gggl_2_test_predictions.nc',
            #                 )
            #             ),
            #         ]
            #     },
            # }

            multi_rare_evaluation(
                pred_paths_per_alpha=pred_paths_per_alpha,
                processes=args.processes,
                two_bin_eval=args.two_bin,
            )

        else:

            print('No prediction_file specified. Creating default paper figure..')
            pred_paths = main_pred_paths(path_prefix=args.path_prefix)
            output_prefix = '0.0'
            histogram = True

            # Load only one file per model for testing
            # TODO: REMOVE AFTER TESTING
            # pred_paths = {k: v[:1] for k, v in pred_paths.items()}
            # pred_paths = {'ConvMOS': pred_paths['ConvMOS'][:1]}
            # pred_paths = {'ConvMOS': pred_paths['ConvMOS']}
            # histogram = False

            multi_evaluation(
                pred_paths,
                processes=args.processes,
                output_prefix=output_prefix,
                histogram=histogram,
            )

    else:

        evaluate(args.prediction_file, args.seasonal, args.log_norm)

    # data = np.random.rand(121, 121)
    # plot_map(data, data.min(), data.max(), title='test', no_colorbar=False)
