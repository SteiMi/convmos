from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr


def evaluate(nc_file: str):
    data = xr.open_dataset(nc_file).load()
    print(data)

    mean_data = data['tg'].mean('time')

    mean_data.plot()
    plt.show()

    print(mean_data)

    land_mask = np.invert(np.isnan(mean_data.values))
    plt.imshow(land_mask)
    plt.show()

    np.save('remo_eobs_land_mask.npy', land_mask)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        'nc_file', type=str, help='Path to a E-OBS temperature .nc-file'
    )
    args = parser.parse_args()

    evaluate(args.nc_file)
