#!/usr/bin/env python3
"""Run RQ2 deep learning analysis.

Research Question 2 evaluates the extent to which LLM enhancement of
human-crafted phishing emails impacts the discriminability of deep learning
models as a model family.

Primary metric: SDT-based sensitivity, d-prime (d').
Complementary metric: AUC-ROC.

The script trains LSTM, GRU, and CNN text classifiers using tokenized body text.
Splits are grouped by template family (--split grouped, primary analysis) or
stratified at random (--split random, secondary analysis); outputs go to
output/<split>/. For each split, models are trained on legitimate emails plus original
phishing emails. The baseline test set uses legitimate emails plus original
phishing emails. The enhanced test set uses the same legitimate emails and the
LLM-enhanced versions of the same phishing pair_ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        'PyTorch is required for RQ2. Install a working torch build in this environment before running RQ2.'
    ) from exc


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == 'scripts':
    RQ_DIR = SCRIPT_PATH.parents[1]
else:
    RQ_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = RQ_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from splits import make_split, unit_table  # noqa: E402
from sensitivity import add_sensitivity_args, apply_sensitivity, run_label  # noqa: E402

SENSITIVITY_AUDIT: list[dict] = []

DEFAULT_MASTER = PROJECT_ROOT.parent / 'processed' / 'dissertation_aligned_source_qc' / 'Final_Paired_Email_Dataset.csv'
OUTPUT_DIR = RQ_DIR / 'output'  # set to output/<split> in main()
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize(text: object) -> list[str]:
    return TOKEN_RE.findall('' if pd.isna(text) else str(text).lower())


def build_vocab(texts: pd.Series, max_vocab_size: int, min_freq: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, freq in counts.most_common(max_vocab_size - 2):
        if freq < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def encode_texts(texts: pd.Series, vocab: dict[str, int], max_length: int) -> np.ndarray:
    arr = np.zeros((len(texts), max_length), dtype=np.int64)
    unk = vocab['<UNK>']
    for i, text in enumerate(texts):
        ids = [vocab.get(tok, unk) for tok in tokenize(text)[:max_length]]
        if ids:
            arr[i, :len(ids)] = ids
    return arr


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


class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, kind: str, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        rnn_cls = nn.LSTM if kind == 'lstm' else nn.GRU
        self.rnn = rnn_cls(embedding_dim, hidden_dim, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        # Texts are right-padded with index 0. Packing makes the final hidden
        # state the one after each email's last real token, not after the
        # trailing padding.
        lengths = x.ne(0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last = hidden[-1]
        return self.fc(self.dropout(last)).squeeze(1)


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int, kernel_sizes: list[int], dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            activated = torch.relu(conv(embedded))
            pooled.append(torch.max(activated, dim=2).values)
        features = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(features)).squeeze(1)


def build_model(model_name: str, vocab_size: int, args: argparse.Namespace) -> nn.Module:
    if model_name == 'lstm':
        return RecurrentClassifier(vocab_size, args.embedding_dim, args.hidden_dim, 'lstm', args.dropout)
    if model_name == 'gru':
        return RecurrentClassifier(vocab_size, args.embedding_dim, args.hidden_dim, 'gru', args.dropout)
    if model_name == 'cnn':
        return CNNClassifier(vocab_size, args.embedding_dim, args.num_filters, [3, 4, 5], args.dropout)
    raise ValueError(f'Unknown model: {model_name}')


def model_epochs(args: argparse.Namespace) -> dict[str, int]:
    """Fixed training epochs per model (set from the epoch check before RQ2)."""
    return {
        name: getattr(args, f'epochs_{name}', None) or args.epochs
        for name in ['lstm', 'gru', 'cnn']
    }


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_criterion(y_train: np.ndarray, device: torch.device) -> nn.Module:
    """Weighted BCE: each phishing error counts neg/pos times a legitimate error."""
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    grad_clip: float, device: torch.device) -> float:
    """One pass over the training data; returns the mean training loss."""
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        # Caps the size of each update so a single large gradient cannot
        # derail training (the standard safeguard for recurrent networks).
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(xb)
        count += len(xb)
    return total / max(count, 1)


def train_model(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, epochs: int,
                args: argparse.Namespace, device: torch.device) -> nn.Module:
    model.to(device)
    criterion = make_criterion(y_train, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    for _ in range(epochs):
        train_one_epoch(model, loader, criterion, optimizer, args.grad_clip, device)
    return model


def predict_scores(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    dummy_y = np.zeros(len(x), dtype=np.float32)
    loader = make_loader(x, dummy_y, batch_size, shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


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


def timing_audit(iteration_results: pd.DataFrame) -> list[dict]:
    """Compute time from the per-row timing columns; sums correctly across resumed sessions."""
    if 'iteration_seconds' not in iteration_results.columns:
        return [{'item': 'timed_iterations', 'value': 0}]
    timed = iteration_results.dropna(subset=['iteration_seconds'])
    per_iteration = timed.drop_duplicates('iteration')['iteration_seconds']
    audit = [
        {'item': 'timed_iterations', 'value': int(per_iteration.size)},
        {'item': 'compute_seconds_total', 'value': round(float(per_iteration.sum()), 1)},
        {'item': 'seconds_per_iteration_mean', 'value': round(float(per_iteration.mean()), 2) if per_iteration.size else None},
    ]
    for model, seconds in timed.groupby('model')['model_seconds'].mean().items():
        audit.append({'item': f'seconds_per_model_fit_mean_{model}', 'value': round(float(seconds), 2)})
    return audit


def build_rq2_outputs(iteration_results: pd.DataFrame, args: argparse.Namespace, master_path: Path, legitimate_ids: list[str], phishing_ids: list[str], device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    iteration_results = iteration_results.drop_duplicates(subset=['iteration', 'model'], keep='last')
    iteration_results = iteration_results.sort_values(['iteration', 'model']).reset_index(drop=True)

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
            'scope': 'deep_learning_family',
            'metric': 'd_prime',
            'role': 'primary',
            **corrected_resampled_t_test(
                family_by_iteration['family_d_prime_difference_baseline_minus_enhanced'],
                args.test_size,
            ),
        },
        {
            'scope': 'deep_learning_family',
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
        if len(values) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(values)
            skewness = stats.skew(values, bias=False)
            kurtosis = stats.kurtosis(values, bias=False)
        else:
            shapiro_stat, shapiro_p = float('nan'), float('nan')
            skewness, kurtosis = float('nan'), float('nan')
        normality.append({
            'scope': 'deep_learning_family',
            'metric': metric,
            'role': role,
            'n': len(values),
            'mean_difference': values.mean(),
            'median_difference': values.median(),
            'sd': values.std(ddof=1),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_wilk_statistic': shapiro_stat,
            'shapiro_wilk_p_value': shapiro_p,
        })
    normality_summary = pd.DataFrame(normality)

    audit = pd.DataFrame([
        {'item': 'master_dataset', 'value': str(master_path)},
        {'item': 'master_dataset_sha256', 'value': sha256_file(master_path)},
        {'item': 'legitimate_pair_ids', 'value': len(legitimate_ids)},
        {'item': 'complete_phishing_pair_ids', 'value': len(phishing_ids)},
        {'item': 'split', 'value': args.split},
        {'item': 'requested_iterations', 'value': args.iterations},
        {'item': 'completed_iterations', 'value': family_by_iteration['iteration'].nunique()},
        *timing_audit(iteration_results),
        {'item': 'resume_enabled', 'value': args.resume},
        {'item': 'checkpoint_every', 'value': args.checkpoint_every},
        {'item': 'test_size', 'value': args.test_size},
        {'item': 'train_size', 'value': 1 - args.test_size},
        {'item': 'models', 'value': 'lstm, gru, cnn'},
        {'item': 'tokenization', 'value': "regex tokens [A-Za-z0-9_']+ from body_model"},
        {'item': 'max_vocab_size', 'value': args.max_vocab_size},
        {'item': 'min_freq', 'value': args.min_freq},
        {'item': 'max_length', 'value': args.max_length},
        {'item': 'embedding_dim', 'value': args.embedding_dim},
        {'item': 'hidden_dim', 'value': args.hidden_dim},
        {'item': 'num_filters', 'value': args.num_filters},
        {'item': 'dropout', 'value': args.dropout},
        {'item': 'epochs_per_model', 'value': ', '.join(f'{k}={v}' for k, v in model_epochs(args).items())},
        {'item': 'batch_size', 'value': args.batch_size},
        {'item': 'learning_rate', 'value': args.learning_rate},
        {'item': 'grad_clip_max_norm', 'value': args.grad_clip},
        {'item': 'rnn_input', 'value': 'packed sequences (final state at last real token)'},
        {'item': 'seed', 'value': args.seed},
        {'item': 'device', 'value': str(device)},
        {'item': 'torch_version', 'value': torch.__version__},
        *SENSITIVITY_AUDIT,
    ])

    return family_by_iteration, metric_summary, ttest, normality_summary, audit


def write_rq2_outputs(iteration_results: pd.DataFrame, args: argparse.Namespace, master_path: Path, legitimate_ids: list[str], phishing_ids: list[str], device: torch.device) -> None:
    family_by_iteration, metric_summary, ttest, normality_summary, audit = build_rq2_outputs(
        iteration_results,
        args,
        master_path,
        legitimate_ids,
        phishing_ids,
        device,
    )
    iteration_results.to_csv(OUTPUT_DIR / 'RQ2_iteration_model_results.csv', index=False)
    family_by_iteration.to_csv(OUTPUT_DIR / 'RQ2_family_iteration_results.csv', index=False)
    metric_summary.to_csv(OUTPUT_DIR / 'RQ2_metric_summary.csv', index=False)
    ttest.to_csv(OUTPUT_DIR / 'RQ2_corrected_resampled_t_tests.csv', index=False)
    normality_summary.to_csv(OUTPUT_DIR / 'RQ2_normality_assumption_summary.csv', index=False)
    audit.to_csv(OUTPUT_DIR / 'RQ2_analysis_audit.csv', index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run RQ2 deep learning analysis.')
    parser.add_argument('--master-path', type=Path, default=DEFAULT_MASTER)
    parser.add_argument('--split', choices=['grouped', 'random'], default='grouped',
                        help='grouped = by template family (primary); random = stratified random (secondary).')
    add_sensitivity_args(parser)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--test-size', type=float, default=0.30)
    parser.add_argument('--seed', type=int, default=930)
    parser.add_argument('--max-vocab-size', type=int, default=20000)
    parser.add_argument('--min-freq', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=300)
    parser.add_argument('--embedding-dim', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-filters', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.30)
    parser.add_argument('--epochs', type=int, default=5, help='Epochs for any model without its own --epochs-<model>.')
    # Fixed from the epoch check, run 2 (docs/ANALYSIS_DECISIONS.md A02).
    parser.add_argument('--epochs-lstm', type=int, default=5)
    parser.add_argument('--epochs-gru', type=int, default=3)
    parser.add_argument('--epochs-cnn', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Maximum gradient norm per update (all models).')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--resume', action='store_true', help='Resume from RQ2_iteration_model_results.csv if it exists.')
    parser.add_argument('--checkpoint-every', type=int, default=1, help='Write outputs after this many newly completed iterations when --resume is used.')
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = RQ_DIR / 'output' / run_label(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    iteration_results_path = OUTPUT_DIR / 'RQ2_iteration_model_results.csv'

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA was requested with --device cuda, but torch.cuda.is_available() is False.')

    master, legitimate_ids, phishing_ids = prepare_master(args.master_path, args)
    units = unit_table(master, legitimate_ids, phishing_ids)

    rows = []
    completed_iterations: set[int] = set()
    if args.resume and iteration_results_path.exists():
        existing_results = pd.read_csv(iteration_results_path)
        if {'iteration', 'model'}.issubset(existing_results.columns):
            model_counts = existing_results.groupby('iteration')['model'].nunique()
            completed_iterations = set(model_counts[model_counts.eq(3)].index.astype(int))
            existing_results = existing_results[existing_results['iteration'].isin(completed_iterations)]
            rows = existing_results.to_dict('records')
            print(f'Resuming from {iteration_results_path}')
            print(f'Completed iterations found: {len(completed_iterations)} / {args.iterations}')
        else:
            print(f'Ignoring existing checkpoint without iteration/model columns: {iteration_results_path}')

    newly_completed = 0
    for iteration in range(args.iterations):
        iteration_number = iteration + 1
        if iteration_number in completed_iterations:
            if iteration_number % 25 == 0 or iteration_number == args.iterations:
                print(f'Skipping completed iteration {iteration_number} / {args.iterations}')
            continue

        iteration_start = time.perf_counter()
        split_seed = args.seed + iteration
        set_all_seeds(split_seed)
        train_units, test_units = make_split(units, args.split, args.test_size, split_seed)
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

        vocab = build_vocab(train['body_model'], args.max_vocab_size, args.min_freq)
        x_train = encode_texts(train['body_model'], vocab, args.max_length)
        y_train = train['label_binary'].to_numpy(dtype=np.int64)
        x_baseline = encode_texts(baseline_test['body_model'], vocab, args.max_length)
        y_baseline = baseline_test['label_binary'].to_numpy(dtype=np.int64)
        x_enhanced = encode_texts(enhanced_test['body_model'], vocab, args.max_length)
        y_enhanced = enhanced_test['label_binary'].to_numpy(dtype=np.int64)

        iteration_rows = []
        for model_name in ['lstm', 'gru', 'cnn']:
            model_start = time.perf_counter()
            set_all_seeds(split_seed)
            model = build_model(model_name, len(vocab), args)
            model = train_model(model, x_train, y_train, model_epochs(args)[model_name], args, device)

            baseline_score = predict_scores(model, x_baseline, args.batch_size, device)
            enhanced_score = predict_scores(model, x_enhanced, args.batch_size, device)
            baseline_pred = (baseline_score >= 0.5).astype(int)
            enhanced_pred = (enhanced_score >= 0.5).astype(int)

            baseline_metrics = metric_row(y_baseline, baseline_score, baseline_pred)
            enhanced_metrics = metric_row(y_enhanced, enhanced_score, enhanced_pred)
            row = {'iteration': iteration_number, 'model': model_name}
            for metric, value in baseline_metrics.items():
                row[f'baseline_{metric}'] = value
            for metric, value in enhanced_metrics.items():
                row[f'enhanced_{metric}'] = value
            row['d_prime_difference_baseline_minus_enhanced'] = baseline_metrics['d_prime'] - enhanced_metrics['d_prime']
            row['auc_difference_baseline_minus_enhanced'] = baseline_metrics['roc_auc'] - enhanced_metrics['roc_auc']
            # Wall-clock seconds to train and score this model (timing only; not part of the analysis).
            row['model_seconds'] = time.perf_counter() - model_start
            iteration_rows.append(row)

        # Whole iteration (split, vocabulary, encoding and all three models), stored on each model row.
        iteration_seconds = time.perf_counter() - iteration_start
        for row in iteration_rows:
            row['iteration_seconds'] = iteration_seconds
        rows.extend(iteration_rows)
        newly_completed += 1
        iteration_results = pd.DataFrame(rows)
        iteration_results = iteration_results.drop_duplicates(subset=['iteration', 'model'], keep='last')
        iteration_results = iteration_results.sort_values(['iteration', 'model'])

        if args.resume and (newly_completed % max(args.checkpoint_every, 1) == 0 or iteration_number == args.iterations):
            write_rq2_outputs(iteration_results, args, args.master_path, legitimate_ids, phishing_ids, device)
            print(f'Checkpoint saved through iteration {iteration_number}: {OUTPUT_DIR}')

        if iteration_number % 5 == 0 or iteration_number == args.iterations:
            print(f'Completed iteration {iteration_number} / {args.iterations}')

    iteration_results = pd.DataFrame(rows)
    if iteration_results.empty:
        raise ValueError('No iteration results are available. Check --iterations and --resume settings.')
    iteration_results = iteration_results.drop_duplicates(subset=['iteration', 'model'], keep='last')
    iteration_results = iteration_results.sort_values(['iteration', 'model'])
    write_rq2_outputs(iteration_results, args, args.master_path, legitimate_ids, phishing_ids, device)

    family_by_iteration, metric_summary, ttest, normality_summary, _ = build_rq2_outputs(
        iteration_results,
        args,
        args.master_path,
        legitimate_ids,
        phishing_ids,
        device,
    )

    print('\nRQ2 metric summary')
    print(metric_summary.round(6).to_string(index=False))
    print('\nRQ2 corrected resampled t-tests')
    print(ttest.round(6).to_string(index=False))
    print('\nRQ2 normality summary')
    print(normality_summary.round(6).to_string(index=False))
    print(f'\nCompleted iterations: {family_by_iteration["iteration"].nunique()} / {args.iterations}')
    print(f'Outputs written to: {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
