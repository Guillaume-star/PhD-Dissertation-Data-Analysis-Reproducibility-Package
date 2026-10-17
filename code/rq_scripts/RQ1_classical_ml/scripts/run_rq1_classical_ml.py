#!/usr/bin/env python3
"""Run RQ1 classical machine learning analysis.

Research Question 1 evaluates the extent to which LLM enhancement of
human-crafted phishing emails impacts the discriminability of classical
machine learning models as a model family.

Primary metric: SDT-based sensitivity, d-prime (d').
Complementary metric: AUC-ROC.

The script trains Logistic Regression, Linear SVM, and Random Forest models
on TF-IDF vectorized body text, all with class weights inversely proportional
to class frequency. Splits are grouped by template family (--split grouped,
primary analysis: every template family is entirely in training or entirely
in test) or stratified at random (--split random, secondary analysis). For
each split, models are trained
on legitimate emails plus original phishing emails. The baseline test set uses
legitimate emails plus original phishing emails. The enhanced test set uses the
same legitimate emails and the LLM-enhanced versions of the same phishing
pair_ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RQ_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RQ_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from splits import make_split, unit_table  # noqa: E402
from sensitivity import add_sensitivity_args, apply_sensitivity, run_label  # noqa: E402

SENSITIVITY_AUDIT: list[dict] = []

DEFAULT_MASTER = PROJECT_ROOT.parent / 'processed' / 'dissertation_aligned_source_qc' / 'Final_Paired_Email_Dataset.csv'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def d_prime_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    hits = int(((y_true == 1) & (y_pred == 1)).sum())
    misses = int(((y_true == 1) & (y_pred == 0)).sum())
    false_alarms = int(((y_true == 0) & (y_pred == 1)).sum())
    correct_rejections = int(((y_true == 0) & (y_pred == 0)).sum())

    signal_n = hits + misses
    noise_n = false_alarms + correct_rejections
    hit_rate = (hits + 0.5) / (signal_n + 1.0)
    false_alarm_rate = (false_alarms + 0.5) / (noise_n + 1.0)
    return float(stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate))


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


def build_models(seed: int, max_features: int, rf_estimators: int) -> dict[str, Pipeline]:
    tfidf = {
        'lowercase': True,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_features': max_features,
    }
    return {
        'logistic_regression': Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf)),
            ('clf', LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced', random_state=seed)),
        ]),
        'linear_svm': Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf)),
            ('clf', LinearSVC(class_weight='balanced', random_state=seed)),
        ]),
        'random_forest': Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf)),
            ('clf', RandomForestClassifier(
                n_estimators=rf_estimators,
                random_state=seed,
                n_jobs=-1,
                class_weight='balanced_subsample',
            )),
        ]),
    }


def score_model(model: Pipeline, text: pd.Series) -> np.ndarray:
    clf = model.named_steps['clf']
    if hasattr(clf, 'predict_proba'):
        return model.predict_proba(text)[:, 1]
    return model.decision_function(text)


def metric_row(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        'd_prime': d_prime_from_predictions(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_score),
    }


def prepare_master(path: Path, args: argparse.Namespace | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    master = pd.read_csv(path, keep_default_na=False)
    master, SENSITIVITY_AUDIT[:] = apply_sensitivity(master, args)
    required = {'pair_id', 'label', 'label_binary', 'variant', 'body_model', 'template_family_id'}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f'Missing required columns in master dataset: {sorted(missing)}')

    master['body_model'] = master['body_model'].fillna('').astype(str)
    master['label_binary'] = master['label_binary'].astype(int)

    legitimate_ids = master.loc[master['variant'].eq('legitimate'), 'pair_id'].drop_duplicates().tolist()
    phishing_variants = master.loc[master['label'].eq('phishing')].groupby('pair_id')['variant'].agg(set)
    complete_phishing_ids = [
        pair_id for pair_id, variants in phishing_variants.items()
        if {'original_phishing', 'enhanced_phishing'}.issubset(variants)
    ]

    if not legitimate_ids:
        raise ValueError('No legitimate pair_ids found.')
    if not complete_phishing_ids:
        raise ValueError('No complete original/enhanced phishing pairs found.')

    return master, legitimate_ids, complete_phishing_ids


def main() -> None:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description='Run RQ1 classical ML analysis.')
    parser.add_argument('--master-path', type=Path, default=DEFAULT_MASTER)
    parser.add_argument('--split', choices=['grouped', 'random'], default='grouped',
                        help='grouped = by template family (primary); random = stratified random (secondary).')
    add_sensitivity_args(parser)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--test-size', type=float, default=0.30)
    parser.add_argument('--seed', type=int, default=930)
    parser.add_argument('--max-features', type=int, default=50000)
    parser.add_argument('--rf-estimators', type=int, default=100)
    args = parser.parse_args()

    output_dir = RQ_DIR / 'output' / run_label(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    master, legitimate_ids, phishing_ids = prepare_master(args.master_path, args)
    units = unit_table(master, legitimate_ids, phishing_ids)

    rows = []
    for iteration in range(args.iterations):
        train_units, test_units = make_split(units, args.split, args.test_size, args.seed + iteration)
        train_legit_ids = set(train_units.loc[train_units['label_binary'].eq(0), 'pair_id'])
        train_phish_ids = set(train_units.loc[train_units['label_binary'].eq(1), 'pair_id'])
        test_legit_ids = set(test_units.loc[test_units['label_binary'].eq(0), 'pair_id'])
        test_phish_ids = set(test_units.loc[test_units['label_binary'].eq(1), 'pair_id'])

        train = pd.concat([
            master[master['pair_id'].isin(train_legit_ids) & master['variant'].eq('legitimate')],
            master[master['pair_id'].isin(train_phish_ids) & master['variant'].eq('original_phishing')],
        ], ignore_index=True)
        baseline_test = pd.concat([
            master[master['pair_id'].isin(test_legit_ids) & master['variant'].eq('legitimate')],
            master[master['pair_id'].isin(test_phish_ids) & master['variant'].eq('original_phishing')],
        ], ignore_index=True)
        enhanced_test = pd.concat([
            master[master['pair_id'].isin(test_legit_ids) & master['variant'].eq('legitimate')],
            master[master['pair_id'].isin(test_phish_ids) & master['variant'].eq('enhanced_phishing')],
        ], ignore_index=True)

        models = build_models(args.seed + iteration, args.max_features, args.rf_estimators)
        for model_name, model in models.items():
            model.fit(train['body_model'], train['label_binary'])

            baseline_score = score_model(model, baseline_test['body_model'])
            baseline_pred = model.predict(baseline_test['body_model'])
            enhanced_score = score_model(model, enhanced_test['body_model'])
            enhanced_pred = model.predict(enhanced_test['body_model'])

            baseline_metrics = metric_row(baseline_test['label_binary'].to_numpy(), baseline_score, baseline_pred)
            enhanced_metrics = metric_row(enhanced_test['label_binary'].to_numpy(), enhanced_score, enhanced_pred)

            row = {'iteration': iteration + 1, 'model': model_name}
            for metric, value in baseline_metrics.items():
                row[f'baseline_{metric}'] = value
            for metric, value in enhanced_metrics.items():
                row[f'enhanced_{metric}'] = value
            row['d_prime_difference_baseline_minus_enhanced'] = baseline_metrics['d_prime'] - enhanced_metrics['d_prime']
            row['auc_difference_baseline_minus_enhanced'] = baseline_metrics['roc_auc'] - enhanced_metrics['roc_auc']
            rows.append(row)

        if (iteration + 1) % 25 == 0:
            print(f'Completed iteration {iteration + 1} / {args.iterations}')

    iteration_results = pd.DataFrame(rows)
    family_by_iteration = iteration_results.groupby('iteration', as_index=False).agg(
        family_d_prime_difference_baseline_minus_enhanced=('d_prime_difference_baseline_minus_enhanced', 'mean'),
        family_auc_difference_baseline_minus_enhanced=('auc_difference_baseline_minus_enhanced', 'mean'),
    )

    metric_summary_rows = []
    for group_name, group in [('family', iteration_results), *iteration_results.groupby('model')]:
        for metric in ['d_prime', 'roc_auc']:
            baseline_col = f'baseline_{metric}'
            enhanced_col = f'enhanced_{metric}'
            metric_summary_rows.append({
                'group': group_name,
                'metric': metric,
                'baseline_mean': group[baseline_col].mean(),
                'enhanced_mean': group[enhanced_col].mean(),
                'mean_difference_baseline_minus_enhanced': (group[baseline_col] - group[enhanced_col]).mean(),
                'baseline_sd': group[baseline_col].std(ddof=1),
                'enhanced_sd': group[enhanced_col].std(ddof=1),
            })
    metric_summary = pd.DataFrame(metric_summary_rows)

    ttest = pd.DataFrame([
        {
            'scope': 'classical_ml_family',
            'metric': 'd_prime',
            'role': 'primary',
            **corrected_resampled_t_test(
                family_by_iteration['family_d_prime_difference_baseline_minus_enhanced'],
                args.test_size,
            ),
        },
        {
            'scope': 'classical_ml_family',
            'metric': 'roc_auc',
            'role': 'complementary',
            **corrected_resampled_t_test(
                family_by_iteration['family_auc_difference_baseline_minus_enhanced'],
                args.test_size,
            ),
        },
    ])

    normality = []
    for col, metric, role in [
        ('family_d_prime_difference_baseline_minus_enhanced', 'd_prime', 'primary'),
        ('family_auc_difference_baseline_minus_enhanced', 'roc_auc', 'complementary'),
    ]:
        values = family_by_iteration[col].dropna().astype(float)
        shapiro_stat, shapiro_p = stats.shapiro(values)
        normality.append({
            'scope': 'classical_ml_family',
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
        })
    normality_summary = pd.DataFrame(normality)

    elapsed_seconds = time.perf_counter() - start_time
    audit = pd.DataFrame([
        {'item': 'master_dataset', 'value': str(args.master_path)},
        {'item': 'master_dataset_sha256', 'value': sha256_file(args.master_path)},
        {'item': 'legitimate_pair_ids', 'value': len(legitimate_ids)},
        {'item': 'complete_phishing_pair_ids', 'value': len(phishing_ids)},
        {'item': 'split', 'value': args.split},
        {'item': 'iterations', 'value': args.iterations},
        {'item': 'test_size', 'value': args.test_size},
        {'item': 'train_size', 'value': 1 - args.test_size},
        {'item': 'models', 'value': 'logistic_regression, linear_svm, random_forest'},
        {'item': 'class_weight', 'value': 'balanced (LR, SVM); balanced_subsample (RF)'},
        {'item': 'decision_rule', 'value': 'default cutoff: predict() (probability 0.5; SVM score 0)'},
        {'item': 'tfidf_ngram_range', 'value': '(1, 2)'},
        {'item': 'tfidf_min_df', 'value': 2},
        {'item': 'tfidf_max_features', 'value': args.max_features},
        {'item': 'rf_estimators', 'value': args.rf_estimators},
        {'item': 'seed', 'value': args.seed},
        {'item': 'runtime_seconds', 'value': round(elapsed_seconds, 3)},
        {'item': 'runtime_minutes', 'value': round(elapsed_seconds / 60, 3)},
        *SENSITIVITY_AUDIT,
    ])

    iteration_results.to_csv(output_dir / 'RQ1_iteration_model_results.csv', index=False)
    family_by_iteration.to_csv(output_dir / 'RQ1_family_iteration_results.csv', index=False)
    metric_summary.to_csv(output_dir / 'RQ1_metric_summary.csv', index=False)
    ttest.to_csv(output_dir / 'RQ1_corrected_resampled_t_tests.csv', index=False)
    normality_summary.to_csv(output_dir / 'RQ1_normality_assumption_summary.csv', index=False)
    audit.to_csv(output_dir / 'RQ1_analysis_audit.csv', index=False)

    print('\nRQ1 metric summary')
    print(metric_summary.round(6).to_string(index=False))
    print('\nRQ1 corrected resampled t-tests')
    print(ttest.round(6).to_string(index=False))
    print('\nRQ1 normality summary')
    print(normality_summary.round(6).to_string(index=False))
    print(f'\nRQ1 runtime: {elapsed_seconds:.1f} seconds ({elapsed_seconds / 60:.2f} minutes)')
    print(f'\nOutputs written to: {output_dir}')


if __name__ == '__main__':
    main()
