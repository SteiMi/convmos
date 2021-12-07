import os
import numpy as np

import xarray as xr

import utils


def recursive_mkdir(path):
    split_dir = path.split("/")
    for k in range(len(split_dir)):
        d = "/".join(split_dir[: (k + 1)])
        if (d != '') and (not os.path.exists(d)):
            os.mkdir(d)


class RemoBase(object):
    def __init__(
        self,
        input_data_dir,
        target_data_file,
        years,
        elevation_file=None,
        input_var='APRL',
        target_var='rr',
        aux_features=[],
    ):
        self.input_data_dir = input_data_dir
        self.target_data_file = target_data_file
        self.input_var = input_var
        self.target_var = target_var
        self.years = years
        self.elevation_file = elevation_file
        self.aux_features = aux_features
        self.aux_data = None
        self.read_data()

    def _read_input(self):
        print("data dir", self.input_data_dir)
        fnames = sorted(
            [
                os.path.join(self.input_data_dir, f)
                for f in os.listdir(self.input_data_dir)
                if any([str(y) in f for y in self.years])
            ]
        )
        self.input_data = xr.open_mfdataset(fnames, combine='nested', concat_dim='time')
        self.input_data = self.input_data.reset_coords()
        # At first xarray didn't support .nc files following CF conventions which is why I had to rename variables.
        # The following renames the renamed variables back to the original name again.
        if (
            'time_var' in self.input_data
            and 'lat_var' in self.input_data
            and 'lon_var' in self.input_data
        ):
            self.input_data = self.input_data.rename(
                {'time_var': 'time', 'lat_var': 'lat', 'lon_var': 'lon'}
            )
        self.input_data = self.input_data.set_coords(['time', 'lat', 'lon'])
        self.input_data = self.input_data.set_index(time='time')

    def _read_target(self):
        self.target_data = xr.open_dataset(self.target_data_file)
        # At first xarray didn't support .nc files following CF conventions which is why I had to rename variables.
        # The following renames the renamed variables back to the original name again.
        if (
            'time_var' in self.target_data
            and 'lat_var' in self.target_data
            and 'lon_var' in self.target_data
        ):
            self.target_data = self.target_data.rename(
                {'time_var': 'time', 'lat_var': 'lat', 'lon_var': 'lon'}
            )
        self.target_data = self.target_data.set_index(time='time')
        self.target_data = self.target_data.set_coords(['time', 'lon', 'lat'])
        self.target_data = self.target_data.sel(
            time=slice(str(min(self.years)), str(max(self.years)))
        )

    def _read_elevation(self):
        elev = xr.open_dataset(self.elevation_file)
        self.elev = elev.rename({"FIB": "elev"})

    def _read_aux(self):
        self.aux_data = {}
        for feature_dir in self.aux_features:
            fnames = sorted(
                [
                    os.path.join(feature_dir, f)
                    for f in os.listdir(feature_dir)
                    if any([str(y) in f for y in self.years])
                ]
            )
            print('Loading', feature_dir)
            self.aux_data[feature_dir] = xr.open_mfdataset(
                fnames, combine='nested', concat_dim='time'
            )
            self.aux_data[feature_dir] = self.aux_data[feature_dir].reset_coords()
            # At first xarray didn't support .nc files following CF conventions which is why I had to rename variables.
            # The following renames the renamed variables back to the original name again.
            if (
                'time_var' in self.aux_data[feature_dir]
                and 'lat_var' in self.aux_data[feature_dir]
                and 'lon_var' in self.aux_data[feature_dir]
            ):
                self.aux_data[feature_dir] = self.aux_data[feature_dir].rename(
                    {'time_var': 'time', 'lat_var': 'lat', 'lon_var': 'lon'}
                )
            self.aux_data[feature_dir] = self.aux_data[feature_dir].set_coords(
                ['time', 'lat', 'lon']
            )
            self.aux_data[feature_dir] = self.aux_data[feature_dir].set_index(
                time='time'
            )

    def read_data(self):
        self._read_input()
        self._read_target()
        if self.elevation_file is not None:
            self._read_elevation()
        if len(self.aux_features) > 0:
            self._read_aux()


class RemoSuperRes(RemoBase):
    def __init__(
        self,
        input_data_dir,
        target_data_file,
        years,
        elevation_file,
        input_var='APRL',
        target_var='rr',
        aux_features=[],
    ):
        super(RemoSuperRes, self).__init__(
            input_data_dir,
            target_data_file,
            years,
            elevation_file,
            input_var=input_var,
            target_var=target_var,
            aux_features=aux_features,
        )

    # This name is misleading - it does not necessarily create test data
    # It simply doesn't create patches
    def make_test(self, scale1=None, scale2=None):
        # scale1 and scale2 are unused and only there to keep the API consistent
        Y = self.target_data[self.target_var].values  # [:, :, :, np.newaxis]
        X = self.input_data[self.input_var].values
        aux = None
        if self.aux_data:
            for k, ds in self.aux_data.items():
                var_name = utils.infer_remo_var_name(ds)
                aux_to_add = ds[var_name].values

                # Just like with temperature (below) some aux vars have an additional static dimension which I have
                # to get rid of
                if len(aux_to_add.shape) == 4:
                    aux_to_add = aux_to_add[:, 0, :, :]

                if aux is None:
                    aux = aux_to_add[:, :, :, np.newaxis]
                else:
                    aux = np.concatenate(
                        [aux, aux_to_add[:, :, :, np.newaxis]], axis=-1
                    )

        # Fill missing values in Y
        for t in range(Y.shape[0]):
            # E-OBS precipitation data is nan when there is no rain. We therefore replace all nans with 0.
            if self.target_var in ['rr', 'cp', 'lsp', 'sf']:
                Y[t] = np.nan_to_num(Y[t])
            # For temperature we can use interpolation
            elif self.target_var == 'TEMP2':
                Y[t] = utils.fillmiss(Y[t])

        Y = Y[:, :, :, np.newaxis]

        # Temperature has another dimension "height" with only one level, which I have to dispose
        if len(X.shape) == 4:
            # I assume that this is such a temperature dataset now
            X = X[:, 0, :, :]

        elev_arr = self.elev.elev.values[0]

        times = self.target_data.time.values
        lats = [self.target_data.lat.values for i in range(Y.shape[0])]
        lons = [self.target_data.lon.values for i in range(Y.shape[0])]
        return X, aux, elev_arr, Y, lats, lons, times
