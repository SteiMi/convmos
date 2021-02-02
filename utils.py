from datetime import datetime
import os
import xarray as xr
import numpy as np
import pandas as pd
import scipy.interpolate


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


if __name__ == "__main__":
    f = '/scratch/remo/EUR-44/165/e031001e_c165_200001_remap.nc'

    ds = xr.open_dataset(f)
    print(infer_remo_var_name(ds))
