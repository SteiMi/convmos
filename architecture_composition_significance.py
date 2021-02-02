"""
Calculate significance tests for architecture composition study.
"""

from os.path import expanduser, join
import pandas as pd
from scipy.stats import wilcoxon


def test(
    results: pd.DataFrame, arch_a: str, arch_b: str, column: str, sig_p: float = 0.05
) -> str:
    a = results[results['NN.architecture'] == arch_a][column]
    b = results[results['NN.architecture'] == arch_b][column]

    _, wilc_p = wilcoxon(a, b)
    if wilc_p > sig_p:
        wilc_significant = False
    else:
        wilc_significant = True
    return str(wilc_significant) + ' (p = ' + str(wilc_p) + ')'


if __name__ == '__main__':

    results = pd.read_json(
        expanduser(join('~', 'results', 'sd-next', 'ablation', 'results.json'))
    )

    pd.options.display.float_format = '{:,.3f}'.format
    # Note that metrics called 'test_*' are actually calculated on the validation data of the complete data set here
    print(
        results[
            [
                'NN.architecture',
                'test_RMSE',
                'test_Correlation',
                'test_Skill score',
                'test_$R^2$',
                'test_Bias',
            ]
        ]
        .groupby('NN.architecture')
        .mean()
        .sort_values('test_RMSE')
        .to_latex()
    )

    print(results.columns)
    best_architecture = (
        results.groupby('NN.architecture').mean().sort_values('test_RMSE').iloc[0].name
    )
    second_best_architecture = (
        results.groupby('NN.architecture').mean().sort_values('test_RMSE').iloc[1].name
    )
    print('Best architecture composition:', best_architecture)

    best_bias_architecture = (
        results.groupby('NN.architecture')
        .mean()
        .sort_values('test_Bias', ascending=True)
        .iloc[0]
        .name
    )

    # best vs best skill
    best_skill_architecture = (
        results.groupby('NN.architecture')
        .mean()
        .sort_values('test_Skill score', ascending=False)
        .iloc[0]
        .name
    )

    print(
        'Is '
        + best_architecture
        + '\'s skill score significantly different to best ('
        + best_skill_architecture
        + ')?',
        test(results, best_architecture, best_skill_architecture, 'test_Skill score'),
        '\n',
    )

    print(
        'Is '
        + best_architecture
        + '\'s bias significantly different to best ('
        + best_bias_architecture
        + ')?',
        test(results, best_architecture, best_bias_architecture, 'test_Bias'),
        '\n',
    )

    metrics = [
        'test_RMSE',
        'test_Correlation',
        'test_Skill score',
        'test_$R^2$',
        'test_Bias',
    ]

    # best vs second best
    for col in metrics:
        print(
            'Is the best architecture\'s ('
            + best_architecture
            + ') '
            + col
            + ' significantly different to the second best architecture ('
            + second_best_architecture
            + ')?',
            test(results, best_architecture, second_best_architecture, col),
        )
    print()

    # Best vs best without mixing
    for col in metrics:
        print(
            'Is '
            + best_architecture
            + '\'s '
            + col
            + ' significantly different to best architecture without mixing? (gggg)?',
            test(results, best_architecture, 'gggg', col),
        )
    print()

    # gggl vs ggl
    for col in metrics:
        print(
            'Is gggl\'s ' + col + ' significantly different to ggl?',
            test(results, 'gggl', 'ggl', col),
        )
    print()

    # ggl vs gl
    for col in metrics:
        print(
            'Is ggl\'s ' + col + ' significantly different to gl?',
            test(results, 'ggl', 'gl', col),
        )
    print()
