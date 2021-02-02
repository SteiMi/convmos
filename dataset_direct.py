from os.path import join
from typing import List, Optional

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from config_loader import config
from remo_preprocess import RemoSuperRes


class RemoDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        val: bool = False,
        transform: Optional[object] = None,
        target_transform: Optional[object] = None,
        debug_with_few_samples: bool = False,
        create_standardization_transform: bool = False,
    ) -> None:

        if train and val:
            print('"train" and "val" cannot be true at the same time!')
            raise ValueError

        min_year = config.getint('DataOptions', 'min_year')
        max_year = config.getint('DataOptions', 'max_year')
        max_train_year = config.getint('DataOptions', 'max_train_year')
        max_val_year = config.getint('DataOptions', 'max_val_year')
        remo_input_dir = config.get('Paths', 'remo_input')
        remo_target_dir = config.get('Paths', 'remo_target')
        elev_file = config.get('Paths', 'elevation')
        input_var = config.get('DataOptions', 'input_variable')
        target_var = config.get('DataOptions', 'target_variable')
        aux_base_path = config.get('Paths', 'aux_base_path')
        # The filter removes empty strings in the resulting list, which occur when there are no aux_variables specified
        aux_vars = list(
            filter(None, config.get('DataOptions', 'aux_variables').split(','))
        )

        aux_vars = [join(aux_base_path, p) for p in aux_vars]

        if train:
            self.years = list(range(min_year, max_train_year + 1))
            print('Train years', self.years)
            self.mode = 'train'

        elif val:

            self.years = list(range(max_train_year + 1, max_val_year + 1))
            print('Validation years', self.years)
            self.mode = 'val'

        else:
            self.years = list(range(max_val_year + 1, max_year + 1))
            print('Test years', self.years)
            self.mode = 'test'

        self.transform = transform
        self.target_transform = target_transform

        # For testing purposes
        # self.years = [2000]

        self.dataset = RemoSuperRes(
            remo_input_dir,
            remo_target_dir,
            self.years,
            elev_file,
            input_var=input_var,
            target_var=target_var,
            aux_features=aux_vars,
        )
        # This name is misleading - it does not necessarily create test data
        (
            self.X,
            self.aux,
            self.elev_arr,
            self.Y,
            self.lats,
            self.lons,
            self.times,
        ) = self.dataset.make_test()

        if debug_with_few_samples:
            num_debug_samples = 16
            print(f'DEBUG: Using only {num_debug_samples} samples')
            self.X = self.X[:num_debug_samples]
            if self.aux is not None:
                self.aux = self.aux[:num_debug_samples]
            self.Y = self.Y[:num_debug_samples]
            self.times = self.times[:num_debug_samples]

        # Convert E-OBS temperature from Celsius to Kelvin
        if target_var == 'tg':
            self.Y += 272.15

        if create_standardization_transform:
            self.standardize_transform = self.calculate_standardization_transform()
            if self.transform is not None:
                self.transform = Compose([self.transform, self.standardize_transform])
            else:
                self.transform = self.standardize_transform

    def __getitem__(self, i: int) -> tuple:
        if self.aux is not None:
            x = np.concatenate(
                (
                    self.X[i, :, :, np.newaxis],
                    self.aux[i],
                    self.elev_arr[:, :, np.newaxis],
                ),
                axis=-1,
            )
        else:
            x = np.concatenate(
                (self.X[i, :, :, np.newaxis], self.elev_arr[:, :, np.newaxis]), axis=-1
            )
        y = self.Y[i]

        if self.transform is not None:
            x = self.transform(x)  # type: ignore

        if self.target_transform is not None:
            y = self.target_transform(y)  # type: ignore

        return x, y

    def __len__(self) -> int:
        return self.X.shape[0]

    def calculate_standardization_transform(self) -> Normalize:
        from scipy import stats

        x_stats = stats.describe(self.X, axis=None)
        elev_stats = stats.describe(self.elev_arr, axis=None)

        if self.aux is not None:
            self.aux_means: List[float] = []
            self.aux_stds: List[float] = []
            for i in range(self.aux.shape[-1]):
                aux_stats = stats.describe(self.aux[:, :, :, i], axis=None)
                self.aux_means.append(aux_stats.mean)
                self.aux_stds.append(np.sqrt(aux_stats.variance))

        self.x_mean = x_stats.mean
        self.x_std = np.sqrt(x_stats.variance)
        self.elev_mean = elev_stats.mean
        self.elev_std = np.sqrt(elev_stats.variance)

        if self.aux is not None:
            means = tuple([self.x_mean] + self.aux_means + [self.elev_mean])
            stds = tuple([self.x_std] + self.aux_stds + [self.elev_std])
        else:
            means = tuple([self.x_mean, self.elev_mean])
            stds = tuple([self.x_std, self.elev_std])

        print('Standardization means:', means)
        print('Standardization stds:', stds)

        return Normalize(means, stds)


if __name__ == "__main__":
    ds = RemoDataset(
        train=True, transform=ToTensor(), create_standardization_transform=True
    )
    ds.__getitem__(0)
    print(ds.__len__())
    print(len(ds))

    print(ds.years)
