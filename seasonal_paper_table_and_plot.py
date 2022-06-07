from os.path import isfile, join, expanduser
import pickle
from typing import List, Dict, Optional
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from scipy.stats import wilcoxon
import xarray as xr

from evaluate import calculate_metrics, init_mpl
from utils import fisher_z_mean


def seasonal_plot_grid(
    model_metrics: Dict[str, Dict[str, List[Dict[str, np.array]]]],
    baseline_metrics: Dict[str, Dict[str, np.array]],
    metric: str,
    vmin: float,
    vmax: float,
    models_to_plot: List[str],
):
    width_in_inches = 7.0056039851
    fig = plt.figure(
        figsize=(width_in_inches, 100)
    )  # Height simply has to be large enough due to aspect in imshow

    nrows = 4
    ncols = 3
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        direction='row',
        axes_pad=0.35,
        share_all=True,
        cbar_location='right',
        cbar_mode='single',
        cbar_size='3%',
        cbar_pad=0.15,
    )

    norm = None

    axs = list(grid)

    for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
        im = axs[i * ncols].imshow(
            baseline_metrics[season][metric],
            # np.random.random((10, 10)),
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            extent=(-1.43, 22.22, 57.06, 42.77),
            aspect=1.6545823175,
        )
        if i == 0:
            axs[i * ncols].set_title('REMO raw')
        axs[i * ncols].invert_yaxis()
        axs[i * ncols].yaxis.set_major_locator(plt.MaxNLocator(3))
        axs[i * ncols].xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        axs[i * ncols].yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
        axs[i * ncols].set_ylabel(season, rotation=0, size='large', labelpad=20)
        for j, model in enumerate(models_to_plot):
            ax = axs[i * ncols + j + 1]
            ax.imshow(
                model_metrics[season][model][0][metric],
                # np.random.random((10, 10)),
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                extent=(-1.43, 22.22, 57.06, 42.77),
                aspect=1.6545823175,
            )
            if i == 0:
                ax.set_title(model)
            ax.invert_yaxis()
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    cbar = axs[-2].cax.colorbar(im, extend='max')
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor('face')
    fig.tight_layout()
    plt.savefig(join(f'seasonal_{metric}_maps.pdf'), bbox_inches='tight', dpi=200)
    plt.close(fig=fig)


def seasonal_evaluation(
    path_prefix: str = join('~', 'convmos_rmse_maps'),
    processes: Optional[int] = None,
    yearly: bool = False,
):
    paper_pred_paths = {
        'ConvMOS': [
            expanduser(join(path_prefix, 'convmos_gggl_13_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_1_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_2_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_3_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_4_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_5_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_6_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_7_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_8_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_9_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_10_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_11_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_12_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_14_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_15_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_16_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_17_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_18_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_19_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_gggl_20_test_predictions.nc')),
        ],
        'Lin': [expanduser(join(path_prefix, 'linear_test_predictions.nc'))],
        'NL PCR': [expanduser(join(path_prefix, 'nlpcr_test_predictions.nc'))],
        'NL RF': [
            expanduser(join(path_prefix, 'nlrf_19_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_1_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_2_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_3_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_4_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_5_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_6_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_7_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_8_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_9_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_10_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_11_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_12_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_13_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_14_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_15_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_16_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_17_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_18_test_predictions.nc')),
            expanduser(join(path_prefix, 'nlrf_20_test_predictions.nc')),
        ],
        'U-Net': [
            expanduser(join(path_prefix, 'unet_15_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_1_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_2_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_3_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_4_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_5_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_6_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_7_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_8_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_9_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_10_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_11_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_12_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_13_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_14_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_16_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_17_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_18_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_19_test_predictions.nc')),
            expanduser(join(path_prefix, 'unet_20_test_predictions.nc')),
        ],
        'CM U-Net': [
            expanduser(join(path_prefix, 'convmos_unet_11_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_1_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_2_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_3_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_4_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_5_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_6_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_7_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_8_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_9_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_10_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_12_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_13_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_14_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_15_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_16_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_17_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_18_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_19_test_predictions.nc')),
            expanduser(join(path_prefix, 'convmos_unet_20_test_predictions.nc')),
        ],
        'ResNet18': [
            expanduser(join(path_prefix, 'resnet18_20_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_1_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_2_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_3_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_4_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_5_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_6_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_7_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_8_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_9_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_10_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_11_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_12_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_13_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_14_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_15_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_16_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_17_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_18_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet18_19_test_predictions.nc')),
        ],
        'ResNet34': [
            expanduser(join(path_prefix, 'resnet34_15_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_1_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_2_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_3_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_4_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_5_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_6_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_7_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_8_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_9_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_10_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_11_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_12_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_13_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_14_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_16_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_17_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_18_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_19_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet34_20_test_predictions.nc')),
        ],
        'ResNet50': [
            expanduser(join(path_prefix, 'resnet50_1_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_2_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_3_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_4_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_5_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_6_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_7_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_8_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_9_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_10_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_11_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_12_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_13_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_14_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_15_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_16_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_17_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_18_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_19_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet50_20_test_predictions.nc')),
        ],
        'ResNet101': [
            expanduser(join(path_prefix, 'resnet101_18_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_1_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_2_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_3_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_4_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_5_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_6_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_7_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_8_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_9_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_10_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_11_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_12_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_13_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_14_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_15_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_16_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_17_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_19_test_predictions.nc')),
            expanduser(join(path_prefix, 'resnet101_20_test_predictions.nc')),
        ],
    }

    print('Loading prediction files...')
    paper_preds = {
        k: [xr.open_dataset(path).load() for path in paths]
        for k, paths in paper_pred_paths.items()
    }

    metric = 'RMSE'
    season_model_metrics = {}
    season_baseline_metrics = {}
    seasons = ['DJF', 'JJA', 'MAM', 'SON'] if not yearly else list(range(2011, 2016))
    for season in seasons:

        if yearly:
            season_preds = {
                k: [v.sel(time=v['time.year'] == season) for v in v_l]
                for k, v_l in paper_preds.items()
            }
        else:
            season_preds = {
                k: [v.sel(time=v['time.season'] == season) for v in v_l]
                for k, v_l in paper_preds.items()
            }

        print('Calculating metrics...')
        cached_mm_file = join('eval_out', f'model_metrics_{season}.pkl')

        if isfile(cached_mm_file):
            print(f'Loading cached model metrics from {cached_mm_file}')
            model_metrics = pickle.load(open(cached_mm_file, 'rb'))
        else:
            model_metrics = {
                k: [
                    calculate_metrics(v.pred, v.target, processes=processes)
                    for v in v_l
                ]
                for k, v_l in season_preds.items()
            }
            pickle.dump(model_metrics, open(cached_mm_file, 'wb'))

        # Just use the first best dataset for the baseline data, as it is in every .nc file anyway
        some_dataset = next(iter(season_preds.values()))[0]
        baseline_metrics = calculate_metrics(some_dataset.input, some_dataset.target)

        season_model_metrics[season] = model_metrics
        season_baseline_metrics[season] = baseline_metrics

        condensed_model_metrics = {}
        baseline_mean = np.nanmean(baseline_metrics[metric])
        print(season)
        print(metric)
        print(
            'None',
            '\t',
            baseline_mean,
            # f'({np.nanstd(baseline_metrics[metric])})', # This does not make sense as there is only one climate run
        )
        for model, metrics in model_metrics.items():
            mm = [np.nanmean(m[metric]) for m in metrics]
            if model == 'Lin' or model == 'NL PCR':
                # Lin and NL PCR are deterministic and will always yield the same results.
                # To keep the code consistent and instead of calculating the same metrics 20 times, I will just
                # calculate them once and append them 20 times
                condensed_model_metrics[model] = mm * len(paper_preds['ConvMOS'])
            else:
                condensed_model_metrics[model] = mm
            perc_to_baseline = np.array(mm) / baseline_mean
            if metric == 'Correlation':
                print(
                    model,
                    '\t',
                    fisher_z_mean(np.array(mm)),
                    f'({np.nanstd(mm)})',
                    '\t',
                    f'{fisher_z_mean(perc_to_baseline)}/{np.nanmean(perc_to_baseline)}%',
                    f'{np.nanstd(perc_to_baseline)}%',
                )
            else:
                print(
                    model,
                    '\t',
                    np.nanmean(mm),
                    f'({np.nanstd(mm)})',
                    '\t',
                    f'{np.nanmean(perc_to_baseline)}%',
                    f'{np.nanstd(perc_to_baseline)}%',
                )

        print('\n\n')
        # season_model_metrics[season] = {'bla': [], 'blupp': []}
        # season_baseline_metrics[season] = {'bla': [], 'blupp': []}

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
                        'CM U-Net vs.',
                        model,
                        ', Wilcoxon p:',
                        wilc_p,
                        'significant?',
                        wilc_significant,
                    )

    if not yearly:
        vmin = 0.0
        vmax = 10.0

        models_to_plot = ['ConvMOS', 'NL PCR']
        seasonal_plot_grid(
            season_model_metrics,
            season_baseline_metrics,
            metric,
            vmin,
            vmax,
            models_to_plot,
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
    parser.add_argument('-y', '--yearly', action='store_true', help='Evaluate per Year')
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

    seasonal_evaluation(
        processes=args.processes, path_prefix=args.path_prefix, yearly=args.yearly
    )
