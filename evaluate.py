from os.path import dirname, join, expanduser
from typing import Dict, List, Tuple
from argparse import ArgumentParser
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import ImageGrid
from multiprocessing import Pool
import numpy as np
from scipy.stats import wilcoxon
import xarray as xr

from metrics import combined_nrsme_rmse_mse_pbias_bias_rsq, correlation, skill
from utils import fisher_z_mean


def set_size(
    width: float = 506.295, fraction: float = 1.0, subplot: List[int] = [1, 1]
):
    """Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
            Default value 347.12354 is textwidth for Springer llncs
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    From: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplot[0] / subplot[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def init_mpl(usetex: bool = True):
    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": usetex,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }

    mpl.rcParams.update(nice_fonts)


def plot_map(
    data: np.array,
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


def calculate_metrics_for_cell(
    preds, obs
) -> Tuple[float, float, float, float, float, float, float, float]:
    # Remove data points from both REMO and E-OBS if E-OBS is nan for that day
    nan_mask = ~np.isnan(obs)

    corr_ = correlation(preds[nan_mask], obs[nan_mask])
    nrmse_, rmse_, mse_, pbias_, bias_, rsq_ = combined_nrsme_rmse_mse_pbias_bias_rsq(
        preds[nan_mask], obs[nan_mask]
    )
    skill_ = skill(preds[nan_mask], obs[nan_mask])

    return corr_, nrmse_, rmse_, mse_, pbias_, bias_, rsq_, skill_


def calculate_metrics(preds, obs):

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
    with Pool(processes=None) as pool:
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


def mean_metrics(metrics: dict) -> dict:

    return {
        'Correlation': fisher_z_mean(
            metrics['Correlation'][~np.isnan(metrics['Correlation'])]
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
):

    print(f'Model results (Season {season_name}):')
    model_metrics = calculate_metrics(predictions.pred, predictions.target)
    print(mean_metrics(model_metrics))

    print(f'Baseline results (Season {season_name}):')
    baseline_metrics = calculate_metrics(predictions.input, predictions.target)
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

        if metric_name == '$R^2$' and vmin < -10.0:
            vmin = -10.0

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


def evaluate(prediction_file: str, seasonal: bool = False, log_norm: bool = False):

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
            )

    else:
        eval_and_plot(
            predictions, graph_save_folder=graph_save_folder, log_norm=log_norm
        )


def plot_grid(
    model_metrics: Dict[str, List[Dict[str, np.array]]],
    baseline_metrics: Dict[str, np.str],
    metric: str,
    vmin: float,
    vmax: float,
    suffix: str = '',
):
    # I use 0.95*\textwidth
    width_in_inches = 7.0056039851
    fig = plt.figure(
        figsize=(width_in_inches, 100)
    )  # Height simply has to be large enough due to aspect in imshow

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 3),
        axes_pad=0.35,
        share_all=True,
        cbar_location='right',
        cbar_mode='single',
        cbar_size='5%',
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
    plt.savefig(join(f'{metric}_maps{suffix}.pdf'), bbox_inches='tight', dpi=200)
    plt.close(fig=fig)


def plot_histogram(model_preds: Dict[str, xr.Dataset]):
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

    fig, ax = plt.subplots(figsize=set_size(fraction=0.5))
    ax.hist(
        data,
        label=labels,
        histtype='bar',
        bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
        log=True,
    )
    ax.legend()
    ax.set_xlabel('Precipitation [mm]')
    fig.tight_layout()
    plt.savefig('histogram.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig=fig)


def paper_figure(seasonal: bool = False):
    paper_pred_paths = {
        'ConvMOS': [
            expanduser(
                '~/convmos_rmse_maps/convmos_gggl_test_predictions.nc'
            ),  # this is 13 (gggl)
            expanduser('~/convmos_rmse_maps/convmos_gggl_1_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_2_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_3_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_4_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_5_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_6_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_7_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_8_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_9_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_10_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_11_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_12_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_14_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_15_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_16_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_17_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_18_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_19_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_20_test_predictions.nc'),
        ],
        'Lin': [expanduser('~/convmos_rmse_maps/linear_test_predictions.nc')],
        'NL PCR': [expanduser('~/convmos_rmse_maps/nlpcr_test_predictions.nc')],
        'NL RF': [
            expanduser('~/convmos_rmse_maps/nlrf_test_predictions.nc'),  # nlrf-19
            expanduser('~/convmos_rmse_maps/nlrf_1_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_2_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_3_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_4_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_5_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_6_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_7_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_8_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_9_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_10_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_11_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_12_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_13_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_14_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_15_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_16_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_17_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_18_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/nlrf_20_test_predictions.nc'),
        ],
    }

    print('Loading prediction files...')
    paper_preds = {k: [xr.open_dataset(path).load() for path in paths] for k, paths in paper_pred_paths.items()}

    print('Plot distributions...')
    plot_histogram({k: v_l[0] for k, v_l in paper_preds.items()})

    print('Calculating metrics...')
    model_metrics = {k: [calculate_metrics(v.pred, v.target) for v in v_l] for k, v_l in paper_preds.items()}
    # Just use the first best dataset for the baseline data, as it is in every .nc file anyway
    some_dataset = next(iter(paper_preds.values()))[0]
    baseline_metrics = calculate_metrics(some_dataset.input, some_dataset.target)

    for metric in ['Correlation', 'Bias', 'RMSE', 'Skill score', '$R^2$']:
        condensed_model_metrics = {}
        for model, metrics in model_metrics.items():
            mm = [np.nanmean(m[metric]) for m in metrics]
            if model == 'Lin' or model == 'NL PCR':
                # Lin and NL PCR are deterministic and will always yield the same results.
                # Instead of calculating 20 times the same metrics, I will just calculate them once and append them
                # 20 times
                condensed_model_metrics[model] = mm * 20
            else:
                condensed_model_metrics[model] = mm
            print(model, '\t', np.nanmean(mm),
                  f'({np.nanstd(mm)})')

        # Significance test
        for model in model_metrics.keys():
            if model != 'ConvMOS':
                _, wilc_p = wilcoxon(condensed_model_metrics['ConvMOS'], condensed_model_metrics[model])
                if wilc_p > 0.05:
                    wilc_significant = False
                else:
                    wilc_significant = True
                print(metric, 'ConvMOS vs.', model, ', Wilcoxon p:', wilc_p, 'significant?', wilc_significant)

    vmin = 0.0
    vmax = 10.0
    # model_metrics = {'ConvMOS': [], 'Lin': [], 'NL PCR': [], 'NL RF': []}
    # baseline_metrics = {}

    metric = 'RMSE'
    plot_grid(model_metrics, baseline_metrics, metric, vmin, vmax)


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
        '-s', '--seasonal', action='store_true', help='Evaluate per Season'
    )
    args = parser.parse_args()

    init_mpl()

    if args.prediction_file is None:
        print('No prediction_file specified. Creating default paper figure..')
        paper_figure()

    else:

        evaluate(args.prediction_file, args.seasonal, args.log_norm)

    # data = np.random.rand(121, 121)
    # plot_map(data, data.min(), data.max(), title='test', no_colorbar=False)
