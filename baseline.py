"""
This implements a typical MOS baseline.
"""

from argparse import ArgumentParser
from datetime import datetime
from os import makedirs
from os.path import join
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.utils.fixes import loguniform
from torchvision.transforms import Compose, ToTensor
import xarray as xr

from config_loader import config
from dataset_direct import RemoDataset
from evaluate import calculate_metrics, mean_metrics
from utils import cur_time_string, write_results_file


def get_datasets() -> Tuple[
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    RemoDataset,
    RemoDataset,
    RemoDataset,
]:

    # create_standardization_transform=True lets the Dataset create a torchvision "Normalize" transform based on the
    # mean/std of the training features. It also applies the newly created transform on all __get_item__ calls.
    train_ds = RemoDataset(
        transform=ToTensor(),
        target_transform=ToTensor(),
        train=True,
        val=False,
        debug_with_few_samples=False,
        create_standardization_transform=True,
    )

    # The train dataset created a torchvision "Normalize" transform with the means/stds of the training data.
    # We want to use the same transform to standardize the test features.
    input_transform = Compose([ToTensor(), train_ds.standardize_transform])

    val_ds = RemoDataset(
        transform=input_transform,
        target_transform=ToTensor(),
        train=False,
        val=True,
        debug_with_few_samples=False,
    )

    test_ds = RemoDataset(
        transform=input_transform,
        target_transform=ToTensor(),
        train=False,
        val=False,
        debug_with_few_samples=False,
    )

    X_train = []
    y_train = []
    for i in range(len(train_ds)):
        x, y = train_ds.__getitem__(i)
        X_train.append(x.numpy())
        y_train.append(y.numpy())

    X_val = []
    y_val = []
    for i in range(len(val_ds)):
        x, y = val_ds.__getitem__(i)
        X_val.append(x.numpy())
        y_val.append(y.numpy())

    X_test = []
    y_test = []
    for i in range(len(test_ds)):
        x, y = test_ds.__getitem__(i)
        X_test.append(x.numpy())
        y_test.append(y.numpy())

    return (
        np.stack(X_train),
        np.stack(y_train),
        np.stack(X_val),
        np.stack(y_val),
        np.stack(X_test),
        np.stack(y_test),
        train_ds,
        val_ds,
        test_ds,
    )


def get_train_test_split(
    X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array
):

    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)

    # Create a split that is simply our train-test split
    folds = np.concatenate(
        [np.ones(X_train.shape[0]) * -1, np.zeros(X_val.shape[0])], axis=0
    )
    split = PredefinedSplit(folds)
    return split, X_combined, y_combined


def find_best_k_for_SelectKBest(
    X_train: np.array,
    y_train: np.array,
    X_val: np.array,
    y_val: np.array,
    n_jobs: int = 1,
    max_k: int = 30,
):
    """
    Find k best features based on a univariate regression but also simply search for the best k with the resulting
    multiple linear regression.
    """

    split, X_combined, y_combined = get_train_test_split(X_train, y_train, X_val, y_val)

    k_upper_bound = (
        X_combined.shape[1] + 1 if X_combined.shape[1] + 1 < max_k else max_k
    )

    param_grid = [{'kbest__k': range(1, k_upper_bound)}]

    # Find optimal features
    pipe = Pipeline(
        steps=[
            ('kbest', SelectKBest(score_func=f_regression)),
            ('regress', LinearRegression()),
        ]
    )

    grid = GridSearchCV(
        pipe, param_grid=param_grid, n_jobs=n_jobs, cv=split, refit=False
    )

    grid.fit(X_combined, y_combined)

    return grid.best_params_['kbest__k']


def find_best_model_parameters(
    X_train: np.array,
    y_train: np.array,
    X_val: np.array,
    y_val: np.array,
    model,
    pca: PCA,
    best_k: int,
    n_jobs: int = 1,
    n_iter: int = 60,
) -> dict:

    est = model()

    split, X_combined, y_combined = get_train_test_split(X_train, y_train, X_val, y_val)

    if model == SVR:
        distributions = {
            'model__C': loguniform(1e-1, 1e3),
            'model__gamma': loguniform(1e-4, 1e0),
            'model__kernel': ['poly', 'rbf', 'sigmoid'],
            'model__degree': [1, 2, 3, 4, 5, 6],
            'model__cache_size': [500],
        }

    elif model == RandomForestRegressor:
        distributions = {
            'model__n_estimators': randint(10, 2000),
            'model__max_features': uniform(0.01, 0.99),  # 0.01-1.0
            'model__max_depth': randint(10, 110),
            'model__min_samples_split': randint(2, 10),
            'model__min_samples_leaf': randint(1, 10),
            'model__bootstrap': [True, False],
            'model__n_jobs': [n_jobs],
        }

        # I think its more efficient to parallelize each RF instead of the search as it is possible for a lot of cores
        # to idle when all param sets are done except for one really long running one
        n_jobs = 1

    else:
        print(
            f'HP search for {str(model)} is not implemented. Returning default parameters.'
        )
        return est

    pipe = Pipeline(
        steps=[
            ('kbest', SelectKBest(score_func=f_regression, k=best_k)),
            ('pca', pca),
            ('model', est),
        ]
    )

    search = RandomizedSearchCV(
        pipe, distributions, cv=split, n_iter=n_iter, n_jobs=n_jobs, refit=False
    )

    search.fit(X_combined, y_combined)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(pd.DataFrame(search.cv_results_))

    # Remove 'model__' from keys
    best_params = {k[7:]: v for k, v in search.best_params_.items()}
    # print('best model parameters:', best_params)

    return model(**best_params)


def run(
    model_name: str = 'linear',
    num_jobs: int = 1,
    n_components: float = 0.95,
    non_local_dist: int = 5,
    n_param_sets: int = 20,
    save_dir: str = '.',
):

    X_train, y_train, X_val, y_val, X_test, y_test, _, val_ds, test_ds = get_datasets()

    non_local = 'nonlocal' in model_name

    # Do not use elevation as it would always be static
    # Elevation is always the last predictor
    X_train = X_train[:, :-1]
    X_val = X_val[:, :-1]
    X_test = X_test[:, :-1]

    print(
        cur_time_string(),
        X_train.shape,
        y_train.shape,
        X_val.shape,
        y_val.shape,
        X_test.shape,
        y_test.shape,
    )

    model_dict = {
        'linear': LinearRegression,
        'linear-nonlocal': LinearRegression,
        'rf': RandomForestRegressor,
        'rf-nonlocal': RandomForestRegressor,
        # Note: SVM takes forever on our datasets, which is why we could not evaluate it
        'svm': SVR,
        'svm-nonlocal': SVR,
    }

    # There is one model per cell
    models = []
    # if non_local:
    pcas = []
    for i in range(X_train.shape[-2] * X_train.shape[-1]):
        # models.append(model_dict[model_name](**hparams_dict))
        models.append(model_dict[model_name]())
        pcas.append(PCA(n_components=n_components))  # Sa'adi2017 does it like this

    train_start = datetime.now()
    print('Training started at', train_start)

    y_val_pred = np.empty(y_val.shape)
    y_test_pred = np.empty(y_test.shape)
    best_ks = np.empty((X_train.shape[-2], X_train.shape[-1]))
    best_features = np.empty((X_train.shape[-2], X_train.shape[-1])).tolist()
    for i in range(X_train.shape[-2]):
        print(cur_time_string(), 'Row', i)
        for j in range(X_train.shape[-1]):
            print(cur_time_string(), 'Column', j)
            model = models[i * X_train.shape[-2] + j]

            if non_local:
                dist = non_local_dist

                min_i = i - dist if i - dist >= 0 else 0
                max_i = (
                    i + dist + 1
                    if i + dist + 1 < X_train.shape[-2]
                    else X_train.shape[-2] - 1
                )
                min_j = j - dist if j - dist >= 0 else 0
                max_j = (
                    j + dist + 1
                    if j + dist + 1 < X_train.shape[-1]
                    else X_train.shape[-1] - 1
                )
                X_train_loc = X_train[:, :, min_i:max_i, min_j:max_j].reshape(
                    X_train.shape[0], -1
                )
                X_val_loc = X_val[:, :, min_i:max_i, min_j:max_j].reshape(
                    X_val.shape[0], -1
                )
                X_test_loc = X_test[:, :, min_i:max_i, min_j:max_j].reshape(
                    X_test.shape[0], -1
                )

            else:
                X_train_loc = X_train[:, :, i, j]
                X_val_loc = X_val[:, :, i, j]
                X_test_loc = X_test[:, :, i, j]

            y_train_loc = y_train[:, 0, i, j]
            y_val_loc = y_val[:, 0, i, j]

            # Speed up things by not doing the feature selection and pca for cells where the label is always static
            if y_train_loc.std() == 0.0:
                pipe = Pipeline(steps=[('model', model)])
                best_ks[i, j] = np.nan

                pipe.fit(X_train_loc, y_train_loc)

            else:

                # Select features
                best_k = find_best_k_for_SelectKBest(
                    X_train_loc, y_train_loc, X_val_loc, y_val_loc, n_jobs=num_jobs
                )
                best_ks[i, j] = best_k

                pca = pcas[i * X_train.shape[-2] + j]

                # print(f'Try to find best model_parameters (n_param_sets: {n_param_sets}, n_jobs: {num_jobs})')
                optimal_model = find_best_model_parameters(
                    X_train_loc,
                    y_train_loc,
                    X_val_loc,
                    y_val_loc,
                    model_dict[model_name],
                    pca,
                    best_k,
                    n_jobs=num_jobs,
                    n_iter=n_param_sets,
                )

                models[i * X_train.shape[-2] + j] = optimal_model

                pipe = Pipeline(
                    steps=[
                        ('kbest', SelectKBest(k=best_k, score_func=f_regression)),
                        ('pca', pca),
                        ('model', optimal_model),
                    ]
                )

                pipe.fit(X_train_loc, y_train_loc)

                best_features[i][j] = pipe[0].get_support(indices=True)

            y_val_pred[:, 0, i, j] = pipe.predict(X_val_loc)
            y_test_pred[:, 0, i, j] = pipe.predict(X_test_loc)

            # Cleanup models as I don't really use them thereafter and especially random forests take a lot of RAM
            models[i * X_train.shape[-2] + j] = None

    # print(y_val_pred)
    print(y_val_pred.shape, y_val.shape)
    print(y_test_pred.shape, y_test.shape)

    np.set_printoptions(threshold=sys.maxsize)
    print('best_ks:')
    print(best_ks)
    print(
        np.nanmean(best_ks), np.nanmin(best_ks), np.nanmax(best_ks), np.nanstd(best_ks)
    )
    print('best_features:')
    print(best_features)

    metrics = calculate_metrics(y_val_pred[:, 0], y_val[:, 0])
    val_res = mean_metrics(metrics)
    print('Validation metrics:')
    print(val_res)
    metrics = calculate_metrics(y_test_pred[:, 0], y_test[:, 0])
    test_res = mean_metrics(metrics)
    print('Test metrics:')
    print(test_res)

    train_end = datetime.now()
    print('Training ended at', train_end)
    print('Training duration:', train_end - train_start)

    results = {}
    # Store the config, ...
    results.update(
        {section_name: dict(config[section_name]) for section_name in config.sections()}
    )
    # ... when training started
    results['train_start'] = str(train_start) if train_start is not None else 'UNKNOWN'
    # ... when training ended
    results['train_end'] = str(train_end) if train_end is not None else 'UNKNOWN'
    # ... how long training lasted
    results['train_duration'] = (
        str(train_end - train_start)
        if train_end is not None and train_start is not None
        else 'UNKNOWN'
    )
    # ... the validation metrics that I calculate,
    results.update({f'val_{k}': v for k, v in val_res.items()})
    # ... and the test metrics that I calculate
    results.update({f'test_{k}': v for k, v in test_res.items()})
    write_results_file(join('results', 'results.json'), pd.json_normalize(results))

    val_preds = xr.Dataset(
        {
            'pred': (['time', 'lat', 'lon'], y_val_pred[:, 0]),
            'input': (
                ['time', 'lat', 'lon'],
                val_ds.X,
            ),  # I cannot use x_val directly as it is standardized
            'target': (['time', 'lat', 'lon'], val_ds.Y[:, :, :, 0]),
        },
        coords={
            'time': val_ds.times,
            'lon_var': (('lat', 'lon'), val_ds.lons[0]),
            'lat_var': (('lat', 'lon'), val_ds.lats[0]),
        },
    )

    test_preds = xr.Dataset(
        {
            'pred': (['time', 'lat', 'lon'], y_test_pred[:, 0]),
            'input': (
                ['time', 'lat', 'lon'],
                test_ds.X,
            ),  # I cannot use x_val directly as it is standardized
            'target': (['time', 'lat', 'lon'], test_ds.Y[:, :, :, 0]),
        },
        coords={
            'time': test_ds.times,
            'lon_var': (('lat', 'lon'), test_ds.lons[0]),
            'lat_var': (('lat', 'lon'), test_ds.lats[0]),
        },
    )

    try:
        makedirs(save_dir)
    except FileExistsError:
        # directory already exists
        pass
    val_preds.to_netcdf(join(save_dir, f'val_predictions.nc'))
    test_preds.to_netcdf(join(save_dir, f'test_predictions.nc'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--num_jobs', type=int, default=1, help='Number of jobs (default: 1)'
    )
    parser.add_argument(
        '--non_local_dist',
        type=int,
        default=config.getint('NN', 'non_local_dist'),
        help='Max cell distance to consider (Only relevant in non_local mode)',
    )
    parser.add_argument(
        '--n_components',
        type=float,
        default=config.getfloat('NN', 'n_components'),
        help='Number of components to retain for PCA (Only relevant in non_local mode)',
    )
    parser.add_argument(
        '--n_param_sets',
        type=int,
        default=config.getint('NN', 'n_param_sets'),
        help='Number of parameter sets tried per cell in the hp search',
    )
    parser.add_argument("--model", type=str, default=config.get('NN', 'model'))
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(config.get('NN', 'scratch'), config.get('SD', 'model_name')),
    )

    args = parser.parse_args()

    print(args)

    run(
        model_name=args.model,
        num_jobs=args.num_jobs,
        n_components=args.n_components,
        non_local_dist=args.non_local_dist,
        n_param_sets=args.n_param_sets,
        save_dir=args.save_dir,
    )
