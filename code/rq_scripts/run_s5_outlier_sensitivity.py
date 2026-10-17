#!/usr/bin/env python3
"""S5 sensitivity analysis: outlier rules on the per-iteration differences.

Reads the primary (or another --run-label) RQ1, RQ2 and RQ3 outputs, removes
iterations flagged by each rule, and re-runs the corrected resampled t-test on
the retained values. The variance correction keeps n = all iterations of the
original run, as in the earlier outlier packet:
(1 / n_total) + test_size / train_size.

Rules: 1.5 x IQR (outside Q1 - 1.5 IQR, Q3 + 1.5 IQR) and +/-3 SD.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy import stats

RQ_ROOT = Path(__file__).resolve().parent
SOURCES = {
    'RQ1': ('RQ1_classical_ml', 'RQ1_family_iteration_results.csv', {
        'd_prime': 'family_d_prime_difference_baseline_minus_enhanced',
        'roc_auc': 'family_auc_difference_baseline_minus_enhanced'}),
    'RQ2': ('RQ2_deep_learning', 'RQ2_family_iteration_results.csv', {
        'd_prime': 'family_d_prime_difference_baseline_minus_enhanced',
        'roc_auc': 'family_auc_difference_baseline_minus_enhanced'}),
    'RQ3': ('RQ3_model_family_comparison', 'RQ3_paired_family_change_results.csv', {
        'd_prime': 'd_prime_change_difference_classical_minus_deep_learning',
        'roc_auc': 'auc_change_difference_classical_minus_deep_learning'}),
}


def corrected_test(values: pd.Series, n_total: int, test_size: float) -> dict:
    n = len(values)
    mean, var = float(values.mean()), float(values.var(ddof=1))
    correction = 1.0 / n_total + test_size / (1.0 - test_size)
    se = math.sqrt(correction * var)
    t = mean / se if se > 0 else math.inf
    half = float(stats.t.ppf(0.975, df=n - 1)) * se
    return {'n_retained': n, 'mean_difference': mean, 'sd': math.sqrt(var), 'correction': correction,
            't_statistic': t, 'degrees_of_freedom': n - 1,
            'p_value_two_sided': float(2 * stats.t.sf(abs(t), df=n - 1)),
            'ci95_low': mean - half, 'ci95_high': mean + half}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-label', default='grouped')
    parser.add_argument('--test-size', type=float, default=0.30)
    args = parser.parse_args()

    out_dir = RQ_ROOT / 'S5_outlier_sensitivity' / 'output' / args.run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, flags = [], []
    for rq, (folder, filename, columns) in SOURCES.items():
        data = pd.read_csv(RQ_ROOT / folder / 'output' / args.run_label / filename)
        for metric, column in columns.items():
            values = data[column].astype(float)
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            keep = {
                'none': pd.Series(True, index=values.index),
                '1.5_IQR': values.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr),
                '3_SD': (values - values.mean()).abs() <= 3 * values.std(ddof=1),
            }
            for rule, mask in keep.items():
                rows.append({'rq': rq, 'metric': metric, 'rule': rule, 'n_removed': int((~mask).sum()),
                             **corrected_test(values[mask], len(values), args.test_size)})
                if rule != 'none':
                    flags += [{'rq': rq, 'metric': metric, 'rule': rule, 'iteration': int(i), 'value': float(v)}
                              for i, v in zip(data.loc[~mask, 'iteration'], values[~mask])]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'S5_outlier_sensitivity_summary.csv', index=False)
    pd.DataFrame(flags, columns=['rq', 'metric', 'rule', 'iteration', 'value']).to_csv(
        out_dir / 'S5_removed_iterations.csv', index=False)
    print(summary.round(6).to_string(index=False))
    print(f'\nOutputs written to: {out_dir}')


if __name__ == '__main__':
    main()
