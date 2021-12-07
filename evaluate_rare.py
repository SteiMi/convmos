from os.path import join
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.cm as mplcm
from matplotlib import colors
from matplotlib import pyplot as plt
from utils import set_size


def find_bin_boundaries(
    b_low: int, b_high: int, data: np.ndarray
) -> Tuple[float, float]:
    """
    Given bin boundaries in percentages and data, find the boundaries in y-space.
    Example: b_low = 0, b_high = 10, y ranges from 100 to 200 -> y_low = 100, y_high = 110
    """
    assert (
        b_low >= 0
    ), 'b_low does not seem to be a percentage (b_low: %d, b_high: %d)' % (
        b_low,
        b_high,
    )
    assert (
        b_low <= 100
    ), 'b_low does not seem to be a percentage (b_low: %d, b_high: %d)' % (
        b_low,
        b_high,
    )
    assert (
        b_high >= 0
    ), 'b_high does not seem to be a percentage (b_low: %d, b_high: %d)' % (
        b_low,
        b_high,
    )
    assert (
        b_low <= 100
    ), 'b_high does not seem to be a percentage (b_low: %d, b_high: %d)' % (
        b_low,
        b_high,
    )
    assert (
        b_low < b_high
    ), 'b_low is not smaller than b_high (b_low: %d, b_high: %d)' % (b_low, b_high)
    # y_min = data['y'].min()
    # y_max = data['y'].max()
    y_min = np.min(data)
    y_max = np.max(data)
    y_range = y_max - y_min
    y_low = y_min + (b_low / 100.0) * y_range
    y_high = y_min + (b_high / 100.0) * y_range
    assert y_low >= y_min
    assert y_high <= y_max
    return y_low, y_high


def create_bins(
    test_targets: np.ndarray,
    bin_percentage: int,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], pd.DataFrame]:
    bin_infos = pd.DataFrame(
        columns=['bin_low', 'bin_high', 'y_low', 'y_high', 'count']
    )
    test_bins: Dict[Tuple[int, int], np.ndarray] = {}
    for b in range(0 + bin_percentage, 100 + bin_percentage, bin_percentage):
        b_low = b - bin_percentage
        b_high = b
        y_low, y_high = find_bin_boundaries(
            b_low,
            b_high,
            test_targets
            # b_low, b_high, est_list_per_alpha[list(est_list_per_alpha.keys())[0]][0]
        )

        if b_high == 100:
            # Set y_high to infinity for the last bin. Otherwise the last data point is missing since df['y'] < y_high.
            y_high = np.inf

        test_bins[(b_low, b_high)] = test_targets[
            (test_targets >= y_low) & (test_targets < y_high)
        ]

        # print((b_low, b_high), (y_low, y_high),
        #       len(est_list_per_bin_per_alpha[(b_low, b_high)][some_valid_a][0]), 'elements')
        bin_info = {
            'bin_low': b_low,
            'bin_high': b_high,
            'y_low': y_low,
            'y_high': y_high,
            'count': len(test_bins[(b_low, b_high)]),
        }
        bin_infos = bin_infos.append(bin_info, ignore_index=True)

    # print(bin_infos)

    return test_bins, bin_infos


def create_two_bins(
    test_targets: np.ndarray,
    bin_edge: float,
) -> Tuple[Dict[Tuple[float, float], np.ndarray], pd.DataFrame]:
    # Simply create one bin for extreme values and one for normal values based on a single edge value.
    bin_infos = pd.DataFrame(
        columns=['bin_low', 'bin_high', 'y_low', 'y_high', 'count']
    )
    test_bins: Dict[Tuple[float, float], np.ndarray] = {}
    y_max = np.max(test_targets)
    y_min = np.min(test_targets)
    perc_of_edge = (bin_edge / (y_max - y_min)) * 100.0
    y_low, y_high = find_bin_boundaries(
        0,
        perc_of_edge,
        test_targets
    )
    test_bins[(0, perc_of_edge)] = test_targets[
        (test_targets >= y_low) & (test_targets < y_high)
    ]
    test_bins[(perc_of_edge, np.inf)] = test_targets[
        (test_targets >= y_high) & (test_targets < np.inf)
    ]
    bin_infos = bin_infos.append({
            'bin_low': 0,
            'bin_high': perc_of_edge,
            'y_low': y_low,
            'y_high': y_high,
            'count': len(test_bins[(0, perc_of_edge)]),
    }, ignore_index=True)
    bin_infos = bin_infos.append({
            'bin_low': perc_of_edge,
            'bin_high': 100,
            'y_low': y_high,
            'y_high': np.inf,
            'count': len(test_bins[(perc_of_edge, np.inf)]),
    }, ignore_index=True)

    return test_bins, bin_infos


def bin_to_metric_plot(
    est_list_per_bin_per_distval: Dict[Tuple[int, int], Dict[Any, List[pd.DataFrame]]],
    bin_infos: pd.DataFrame,
    metric: Callable,
    dvals_to_plot: List[Any] = [a / 10 for a in list(range(0, 21, 2))],
    filename_prefix: str = '',
):

    dataset_bin_info = bin_infos.sort_values(by=['count'])

    # Create mappings from bin to bin rank and vice versa
    bin_to_rank = {}
    rank_to_bin = {}
    for i, bin_row in enumerate(dataset_bin_info.iterrows()):
        bin_to_rank[(bin_row[1]['bin_low'], bin_row[1]['bin_high'])] = i
        rank_to_bin[i] = (bin_row[1]['bin_low'], bin_row[1]['bin_high'])

    fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=set_size(fraction=0.5, subplot=[2, 1])
    )
    axs[0].set_title(filename_prefix)
    axs[1].set_xlabel('Bin Rank')
    if filename_prefix in ['normal', 'dnormal']:
        axs[0].set_ylabel('$p$')
        axs[1].set_ylabel('RMSE')

    axs[0].set_ylim([0, 0.044])
    axs[1].set_ylim([3, 28])

    # Calculate density per bin, see https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/histograms.py#L670-L921
    freq_per_bin = [
        len(b[list(b.keys())[0]][0]) for b in est_list_per_bin_per_distval.values()
    ]
    min_y = min(
        [
            b[list(b.keys())[0]][0]['y'].min()
            for b in est_list_per_bin_per_distval.values()
        ]
    )
    max_y = max(
        [
            b[list(b.keys())[0]][0]['y'].max()
            for b in est_list_per_bin_per_distval.values()
        ]
    )
    bin_size = (max_y - min_y) / len(est_list_per_bin_per_distval.keys())

    xticks = [
        '%d' % int(bin_to_rank[br] + 1) for br in est_list_per_bin_per_distval.keys()
    ]
    axs[0].bar(xticks, freq_per_bin / bin_size / np.sum(freq_per_bin))

    plot_dict: Dict[Any, List[float]] = {}
    for dval in dvals_to_plot:
        plot_dict[dval] = []

    for b_range, dval_dict in est_list_per_bin_per_distval.items():
        for dval, dfs in dval_dict.items():
            if dval in dvals_to_plot:
                scores_in_bin = [metric(df['y'], df['estimate']) for df in dfs]
                plot_dict[dval].append(np.mean(scores_in_bin))

    max_a = 2.0
    cm = plt.get_cmap('viridis')
    cNorm = colors.Normalize(vmin=0, vmax=max_a)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    axs[1].set_prop_cycle(color=[scalarMap.to_rgba(a) for a in dvals_to_plot])
    linestyles = ['-', '--', '-.', ':']
    i = 0
    for dval, metric_vals in plot_dict.items():
        axs[1].plot(
            xticks,
            metric_vals,
            label=dval,
            linestyle=linestyles[i % len(linestyles)],
        )
        i += 1

    metric_str = metric.__name__
    filename = 'bin_to_%s.pdf' % metric_str
    fig.savefig(join('results', filename), format='pdf', bbox_inches='tight')

    figlegend = plt.figure(figsize=(374.0 / 72.27, 1))
    figlegend.legend(
        *axs[1].get_legend_handles_labels(),
        loc='upper center',
        ncol=6,
        fontsize='x-small',
    )
    figlegend.savefig(
        join('results', 'legend_' + filename), format='pdf', bbox_inches='tight'
    )


def extremely_specific_pred_paths_that_I_need_for_rare_evaluation(
    path_prefix: str = join('/', 'scratch', 'steininger', 'deepsd')
) -> Dict[str, Dict[str, List[str]]]:
    # The first .nc file of each model's list bears special significance as this one is used for plotting maps.
    # I always use the best model instance for each model as the first one to show best possible results.
    pred_paths = {
        '0.0': {
            'ConvMOS': [
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_13_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_1_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_2_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_3_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_4_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_5_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_6_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_7_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_8_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_9_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_10_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_11_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_12_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_14_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_15_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_16_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_17_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_18_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_19_test_predictions.nc'),
                join(path_prefix, 'convmos_paper_runs', '0.0', 'convmos_gggl_20_test_predictions.nc'),
            ],
        },
        '1.0': {
            'ConvMOS': [
                # 8 has the best extreme_test_RMSE
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-8',
                    'predictions_test_156.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-1',
                    'predictions_test_75.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-2',
                    'predictions_test_82.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-3',
                    'predictions_test_79.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-4',
                    'predictions_test_102.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-5',
                    'predictions_test_55.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-6',
                    'predictions_test_105.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-7',
                    'predictions_test_134.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-9',
                    'predictions_test_64.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-10',
                    'predictions_test_35.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-11',
                    'predictions_test_66.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-12',
                    'predictions_test_76.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-13',
                    'predictions_test_132.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-14',
                    'predictions_test_104.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-15',
                    'predictions_test_155.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-16',
                    'predictions_test_48.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-17',
                    'predictions_test_148.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-18',
                    'predictions_test_132.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-19',
                    'predictions_test_99.nc',
                ),
                join(
                    path_prefix,
                    'scratch_remo',
                    'denseloss_1.0_ees2',
                    'APRL-rr-11-11-sdnext-convmos-prec-wl-1-0-ees2-20',
                    'predictions_test_66.nc',
                ),
            ]
        },
    }
    return pred_paths
