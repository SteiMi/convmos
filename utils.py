from datetime import datetime
import os
from typing import List
import xarray as xr
import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib as mpl


def set_size(width: float = 338.0, fraction: float = 1.0, subplot: List[int] = [1, 1]):
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


def cur_time_string():
    return datetime.now().strftime('%d-%m-%Y %H:%M:%S')


def infer_remo_var_name(ds: xr.Dataset) -> str:
    """ Infer the variable name via exclusion of known non-vars (like lat, lon etc.) """
    known_non_var_names = [
        'lat',
        'lon',
        'time',
        'time_bnds',
        'plev',
        'height',
        'lat_var',
        'lon_var',
        'time_var',
    ]
    pot_var_names = [str(k) for k in list(ds.variables.keys())]
    remaining_var_names = [n for n in pot_var_names if n not in known_non_var_names]
    if len(remaining_var_names) == 0:
        raise Exception(
            f'Cannot infer REMO var name. All potential var names are excluded. ({str(pot_var_names)})'
        )
    elif len(remaining_var_names) > 1:
        raise Exception(
            f'Cannot infer REMO var name. More than one candidate available. ({str(remaining_var_names)})'
        )

    return remaining_var_names[0]


def fisher_z_mean(arr: np.array) -> float:
    # According to https://link.springer.com/content/pdf/10.1007%2F978-3-642-12770-0.pdf (p. 160)
    zs = 0.5 * np.log((1 + arr) / (1 - arr))
    z_mean = np.mean(zs)
    mean = (np.exp(2 * z_mean) - 1) / (np.exp(2 * z_mean) + 1)
    return mean


def write_results_file(output_file_path: str, results: pd.DataFrame):
    """ Write results into a file either as json or csv """

    file_extension = output_file_path.split('.')[-1]

    print(results)

    results = pd.DataFrame(results, index=[0])

    if file_extension == 'json':
        import portalocker

        # For json we have to deal with the concurrent file access therefore
        # i use portalocker to lock the file during reading, constructing the
        # new json, and writing
        with portalocker.Lock(output_file_path, mode='a+', timeout=120) as f:

            f.seek(0)

            # Read old results file if it exist
            if f.read(1) != '':
                f.seek(0)
                old_results = pd.read_json(f)

                # Delete old content
                f.seek(0)
                f.truncate()

                # Combine old and new results (even if they have different columns)
                results = pd.concat([old_results, results], axis=0, ignore_index=True)

            # Write combined results to file and retry indefinitely if it failed
            results.to_json(f)
            f.flush()
            os.fsync(f.fileno())

    elif file_extension == 'csv':

        # The initial write has to write the column headers if the file doesn't
        # exist yet
        initial_write = not os.path.isfile(output_file_path)

        # Write result to file and retry indefinitely if it failed
        while True:
            try:
                results.to_csv(
                    output_file_path, mode='a', header=initial_write, index=False
                )
            except:
                continue
            break

    else:
        print('Invalid file extension: ', file_extension)


def fillmiss(x):
    """
    I borrowed this from DeepSD's Code.
    """
    if x.ndim != 2:
        raise ValueError("X have only 2 dimensions.")
    mask = ~np.isnan(x)
    xx, yy = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T
    data0 = np.ravel(x[mask])
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    return result0


def flatten_dict(d, c=[]):
    """
    Flattens first two "layers" of a dict.
    """
    flat_dict = {}
    for k1, v1 in d.items():
        for k2, v2 in v1.items():
            flat_dict[(k1, k2)] = v2
    return flat_dict


if __name__ == "__main__":
    f = '/scratch/steininger/deepsd/remo/EUR-44/165/e031001e_c165_200001_remap.nc'

    ds = xr.open_dataset(f)
    print(infer_remo_var_name(ds))
