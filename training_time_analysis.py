"""
Calculate significance tests for architecture composition study.
"""

from os.path import expanduser, join
import pandas as pd
import numpy as np


if __name__ == '__main__':

    results = pd.read_json(
        expanduser(join('~', 'results', 'sd-next', 'time_tests', 'results.json'))
    )

    # Check whether we have results for 5 runs
    print(results.groupby(['NN.model', 'NN.global_module'], dropna=False).size())
    # We aggregate over NN.global_module to differentiate ConvMOS and CM-UNet. Do not get confused with it saying
    # "GlobalNet" in ResNet and UNet baselines - the field doesn't do anything for the baselines and I just didn't
    # change it.
    assert all(
        results.groupby(['NN.model', 'NN.global_module'], dropna=False).size() == 5
    )

    pd.options.display.float_format = '{:,.2f}'.format

    results['train_duration'] = pd.to_timedelta(results['train_duration'])
    results['train_duration_int64'] = results['train_duration'].values.astype(np.int64)

    mean = results[['NN.model', 'NN.global_module', 'train_duration']].groupby(
        ['NN.model', 'NN.global_module'], dropna=False
    ).mean(numeric_only=False) / pd.to_timedelta(1, unit='h')

    print('Mean training times in hours:')
    print(mean)

    std = (
        results[['NN.model', 'NN.global_module', 'train_duration_int64']]
        .groupby(['NN.model', 'NN.global_module'], dropna=False)
        .std()
    )

    std['train_duration_std'] = pd.to_timedelta(
        std['train_duration_int64']
    ) / pd.to_timedelta(1, unit='h')

    print('Standard deviation of training times in hours:')
    print(std)
