from os.path import dirname, join, expanduser
from typing import List, Dict
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from scipy.stats import wilcoxon
import xarray as xr

from evaluate import calculate_metrics, init_mpl


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


def seasonal_evaluation():
    paper_pred_paths = {
        'ConvMOS': [
            expanduser(
                '~/convmos_rmse_maps/convmos_gggl_test_predictions.nc'
            ),  # this is 12 (gggl)
            expanduser('~/convmos_rmse_maps/convmos_gggl_0_test_predictions.nc'),
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
            expanduser('~/convmos_rmse_maps/convmos_gggl_13_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_14_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_15_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_16_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_17_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_18_test_predictions.nc'),
            expanduser('~/convmos_rmse_maps/convmos_gggl_19_test_predictions.nc'),
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
    paper_preds = {
        k: [xr.open_dataset(path).load() for path in paths]
        for k, paths in paper_pred_paths.items()
    }

    metric = 'RMSE'
    season_model_metrics = {}
    season_baseline_metrics = {}
    for season in ['DJF', 'JJA', 'MAM', 'SON']:

        season_preds = {
            k: [v.sel(time=v['time.season'] == season) for v in v_l]
            for k, v_l in paper_preds.items()
        }

        print('Calculating metrics...')
        model_metrics = {
            k: [calculate_metrics(v.pred, v.target) for v in v_l]
            for k, v_l in season_preds.items()
        }
        # Just use the first best dataset for the baseline data, as it is in every .nc file anyway
        some_dataset = next(iter(season_preds.values()))[0]
        baseline_metrics = calculate_metrics(some_dataset.input, some_dataset.target)

        season_model_metrics[season] = model_metrics
        season_baseline_metrics[season] = baseline_metrics

        condensed_model_metrics = {}
        print(season)
        print(metric)
        print(
            'None',
            '\t',
            np.nanmean(baseline_metrics[metric]),
            f'({np.nanstd(baseline_metrics[metric])})',
        )
        for model, metrics in model_metrics.items():
            mm = [np.nanmean(m[metric]) for m in metrics]
            if model == 'Lin' or model == 'NL PCR':
                # Lin and NL PCR are deterministic and will always yield the same results.
                # To keep the code consistent and instead of calculating the same metrics 20 times, I will just
                # calculate them once and append them 20 times
                condensed_model_metrics[model] = mm * 20
            else:
                condensed_model_metrics[model] = mm
            print(model, '\t', np.nanmean(mm), f'({np.nanstd(mm)})')

        print('\n\n')
        # season_model_metrics[season] = {'bla': [], 'blupp': []}
        # season_baseline_metrics[season] = {'bla': [], 'blupp': []}

        for model in model_metrics.keys():
            if model != 'ConvMOS':
                _, wilc_p = wilcoxon(
                    condensed_model_metrics['ConvMOS'], condensed_model_metrics[model]
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
    parser.add_argument(
        '-s', '--seasonal', action='store_true', help='Evaluate per Season'
    )
    args = parser.parse_args()

    init_mpl()

    seasonal_evaluation()
