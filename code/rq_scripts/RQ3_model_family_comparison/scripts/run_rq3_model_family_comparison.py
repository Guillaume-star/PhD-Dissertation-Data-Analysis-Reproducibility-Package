#!/usr/bin/env python3
"""Run RQ3 model-family comparison analysis.

Research Question 3 evaluates whether the impact of LLM enhancement on model
 discriminability differs between classical machine learning and deep learning
model families.

This script does not retrain models. It uses the family-level iteration results
from RQ1 and RQ2. For each repeated split, the family-level change in
discriminability is compared between classical ML and deep learning.

Primary metric: SDT-based sensitivity, d-prime (d').
Complementary metric: AUC-ROC.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import pandas as pd
from scipy import stats


RQ_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RQ_DIR.parent
RQ1_OUTPUT = PROJECT_ROOT / 'RQ1_classical_ml' / 'output'
RQ2_OUTPUT = PROJECT_ROOT / 'RQ2_deep_learning' / 'output'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def corrected_resampled_t_test(differences: pd.Series, test_size: float) -> dict[str, float]:
    values = differences.dropna().astype(float)
    n = len(values)
    mean_diff = float(values.mean())
    sample_var = float(values.var(ddof=1))
    train_size = 1.0 - test_size
    correction = (1.0 / n) + (test_size / train_size)

    if n < 2 or sample_var == 0:
        return {
            'n_observations': n,
            'mean_difference': mean_diff,
            'sample_variance': sample_var,
            'correction': correction,
            't_statistic': math.inf if mean_diff != 0 else 0.0,
            'degrees_of_freedom': n - 1,
            'p_value_two_sided': 0.0 if mean_diff != 0 else 1.0,
            'ci95_low': mean_diff,
            'ci95_high': mean_diff,
        }

    standard_error = math.sqrt(correction * sample_var)
    t_statistic = mean_diff / standard_error
    p_value = float(2 * stats.t.sf(abs(t_statistic), df=n - 1))
    # 95% CI of the mean difference with the same corrected variance.
    half_width = float(stats.t.ppf(0.975, df=n - 1)) * standard_error
    return {
        'n_observations': n,
        'mean_difference': mean_diff,
        'sample_variance': sample_var,
        'correction': correction,
        't_statistic': t_statistic,
        'degrees_of_freedom': n - 1,
        'p_value_two_sided': p_value,
        'ci95_low': mean_diff - half_width,
        'ci95_high': mean_diff + half_width,
    }


def normality_row(values: pd.Series, metric: str, role: str) -> dict[str, float | str | int]:
    values = values.dropna().astype(float)
    shapiro_stat, shapiro_p = stats.shapiro(values)
    return {
        'scope': 'classical_minus_deep_learning_family_change',
        'metric': metric,
        'role': role,
        'n': len(values),
        'mean_difference': values.mean(),
        'median_difference': values.median(),
        'sd': values.std(ddof=1),
        'skewness': stats.skew(values, bias=False),
        'kurtosis': stats.kurtosis(values, bias=False),
        'shapiro_wilk_statistic': shapiro_stat,
        'shapiro_wilk_p_value': shapiro_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run RQ3 model-family comparison analysis.')
    parser.add_argument('--split', choices=['grouped', 'random'], default='grouped',
                        help='Which RQ1/RQ2 run to compare; both must use the same split.')
    parser.add_argument('--run-label', default=None,
                        help='RQ1/RQ2 output subfolder to compare (e.g. S2_masked); defaults to --split.')
    parser.add_argument('--rq1-family-path', type=Path, default=None)
    parser.add_argument('--rq2-family-path', type=Path, default=None)
    parser.add_argument('--test-size', type=float, default=0.30)
    args = parser.parse_args()

    label = args.run_label or args.split
    args.rq1_family_path = args.rq1_family_path or RQ1_OUTPUT / label / 'RQ1_family_iteration_results.csv'
    args.rq2_family_path = args.rq2_family_path or RQ2_OUTPUT / label / 'RQ2_family_iteration_results.csv'
    output_dir = RQ_DIR / 'output' / label
    output_dir.mkdir(parents=True, exist_ok=True)

    rq1 = pd.read_csv(args.rq1_family_path)
    rq2 = pd.read_csv(args.rq2_family_path)

    required_rq1 = {
        'iteration',
        'family_d_prime_difference_baseline_minus_enhanced',
        'family_auc_difference_baseline_minus_enhanced',
    }
    required_rq2 = required_rq1.copy()
    missing_rq1 = required_rq1 - set(rq1.columns)
    missing_rq2 = required_rq2 - set(rq2.columns)
    if missing_rq1:
        raise ValueError(f'Missing columns in RQ1 family results: {sorted(missing_rq1)}')
    if missing_rq2:
        raise ValueError(f'Missing columns in RQ2 family results: {sorted(missing_rq2)}')

    rq1 = rq1.rename(columns={
        'family_d_prime_difference_baseline_minus_enhanced': 'classical_d_prime_change_baseline_minus_enhanced',
        'family_auc_difference_baseline_minus_enhanced': 'classical_auc_change_baseline_minus_enhanced',
    })
    rq2 = rq2.rename(columns={
        'family_d_prime_difference_baseline_minus_enhanced': 'deep_learning_d_prime_change_baseline_minus_enhanced',
        'family_auc_difference_baseline_minus_enhanced': 'deep_learning_auc_change_baseline_minus_enhanced',
    })

    paired = rq1[[
        'iteration',
        'classical_d_prime_change_baseline_minus_enhanced',
        'classical_auc_change_baseline_minus_enhanced',
    ]].merge(
        rq2[[
            'iteration',
            'deep_learning_d_prime_change_baseline_minus_enhanced',
            'deep_learning_auc_change_baseline_minus_enhanced',
        ]],
        on='iteration',
        how='inner',
        validate='one_to_one',
    )

    if paired.empty:
        raise ValueError('No paired iterations were found between RQ1 and RQ2 outputs.')

    paired['d_prime_change_difference_classical_minus_deep_learning'] = (
        paired['classical_d_prime_change_baseline_minus_enhanced']
        - paired['deep_learning_d_prime_change_baseline_minus_enhanced']
    )
    paired['auc_change_difference_classical_minus_deep_learning'] = (
        paired['classical_auc_change_baseline_minus_enhanced']
        - paired['deep_learning_auc_change_baseline_minus_enhanced']
    )

    summary = pd.DataFrame([
        {
            'metric': 'd_prime',
            'role': 'primary',
            'classical_mean_change_baseline_minus_enhanced': paired['classical_d_prime_change_baseline_minus_enhanced'].mean(),
            'deep_learning_mean_change_baseline_minus_enhanced': paired['deep_learning_d_prime_change_baseline_minus_enhanced'].mean(),
            'mean_difference_classical_minus_deep_learning': paired['d_prime_change_difference_classical_minus_deep_learning'].mean(),
            'classical_sd': paired['classical_d_prime_change_baseline_minus_enhanced'].std(ddof=1),
            'deep_learning_sd': paired['deep_learning_d_prime_change_baseline_minus_enhanced'].std(ddof=1),
            'difference_sd': paired['d_prime_change_difference_classical_minus_deep_learning'].std(ddof=1),
        },
        {
            'metric': 'roc_auc',
            'role': 'complementary',
            'classical_mean_change_baseline_minus_enhanced': paired['classical_auc_change_baseline_minus_enhanced'].mean(),
            'deep_learning_mean_change_baseline_minus_enhanced': paired['deep_learning_auc_change_baseline_minus_enhanced'].mean(),
            'mean_difference_classical_minus_deep_learning': paired['auc_change_difference_classical_minus_deep_learning'].mean(),
            'classical_sd': paired['classical_auc_change_baseline_minus_enhanced'].std(ddof=1),
            'deep_learning_sd': paired['deep_learning_auc_change_baseline_minus_enhanced'].std(ddof=1),
            'difference_sd': paired['auc_change_difference_classical_minus_deep_learning'].std(ddof=1),
        },
    ])

    ttest = pd.DataFrame([
        {
            'scope': 'classical_minus_deep_learning_family_change',
            'metric': 'd_prime',
            'role': 'primary',
            **corrected_resampled_t_test(
                paired['d_prime_change_difference_classical_minus_deep_learning'],
                args.test_size,
            ),
        },
        {
            'scope': 'classical_minus_deep_learning_family_change',
            'metric': 'roc_auc',
            'role': 'complementary',
            **corrected_resampled_t_test(
                paired['auc_change_difference_classical_minus_deep_learning'],
                args.test_size,
            ),
        },
    ])

    normality = pd.DataFrame([
        normality_row(paired['d_prime_change_difference_classical_minus_deep_learning'], 'd_prime', 'primary'),
        normality_row(paired['auc_change_difference_classical_minus_deep_learning'], 'roc_auc', 'complementary'),
    ])

    audit = pd.DataFrame([
        {'item': 'rq1_family_results', 'value': str(args.rq1_family_path)},
        {'item': 'rq1_family_results_sha256', 'value': sha256_file(args.rq1_family_path)},
        {'item': 'rq2_family_results', 'value': str(args.rq2_family_path)},
        {'item': 'rq2_family_results_sha256', 'value': sha256_file(args.rq2_family_path)},
        {'item': 'split', 'value': args.split},
        {'item': 'run_label', 'value': label},
        {'item': 'paired_iterations', 'value': len(paired)},
        {'item': 'test_size', 'value': args.test_size},
        {'item': 'train_size', 'value': 1 - args.test_size},
        {'item': 'comparison', 'value': 'classical_ml_family_change_minus_deep_learning_family_change'},
        {'item': 'primary_metric', 'value': 'd_prime'},
        {'item': 'complementary_metric', 'value': 'roc_auc'},
    ])

    paired.to_csv(output_dir / 'RQ3_paired_family_change_results.csv', index=False)
    summary.to_csv(output_dir / 'RQ3_family_change_summary.csv', index=False)
    ttest.to_csv(output_dir / 'RQ3_corrected_resampled_t_tests.csv', index=False)
    normality.to_csv(output_dir / 'RQ3_normality_assumption_summary.csv', index=False)
    audit.to_csv(output_dir / 'RQ3_analysis_audit.csv', index=False)

    print('\nRQ3 family change summary')
    print(summary.round(6).to_string(index=False))
    print('\nRQ3 corrected resampled t-tests')
    print(ttest.round(6).to_string(index=False))
    print('\nRQ3 normality summary')
    print(normality.round(6).to_string(index=False))
    print(f'\nOutputs written to: {output_dir}')


if __name__ == '__main__':
    main()
