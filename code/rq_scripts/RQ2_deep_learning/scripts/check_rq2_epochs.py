#!/usr/bin/env python3
"""One-time learning-curve check for the RQ2 epoch count (run before RQ2).

Uses only the TRAINING portion of the first primary split (grouped by template
family, seed 930). From it, 15% of emails (again whole template families) are
held out as a validation set. Each RQ2 model (LSTM, GRU, CNN) is trained with
the exact RQ2 settings for --max-epochs epochs, and after every epoch the
script records training loss, validation loss, validation AUC and validation
d' at the fixed 0.5 cutoff. The RQ2 test sets are never loaded or scored.

Decision rule (version 2, fixed before the second run): each model is trained
in RQ2 for a fixed number of epochs equal to the FIRST epoch whose validation
loss is within 5% of that model's minimum validation loss over --max-epochs.
The first run (version 1 rule: keep 5 epochs) rejected 5 epochs and showed
unstable LSTM/GRU training, which led to packed sequences and gradient
clipping in run_rq2_deep_learning.py before this rule was set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))
from run_rq2_deep_learning import (  # noqa: E402
    build_model, build_vocab, d_prime_from_predictions, encode_texts, make_criterion, make_loader,
    prepare_master, set_all_seeds, sha256_file, train_one_epoch,
)

TOLERANCE = 0.05
from splits import grouped_stratified_split, unit_table  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "epoch_check_v2"


def evaluate(model, x, y, criterion, batch_size, device):
    model.eval()
    losses, scores = [], []
    with torch.no_grad():
        for xb, yb in make_loader(x, y.astype(np.float32), batch_size, shuffle=False):
            logits = model(xb.to(device))
            losses.append(criterion(logits, yb.to(device)).item() * len(xb))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    scores = np.concatenate(scores)
    return sum(losses) / len(x), roc_auc_score(y, scores), d_prime_from_predictions(y, (scores >= 0.5).astype(int))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master-path", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=930)
    # RQ2 defaults, unchanged
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-filters", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master, legitimate_ids, phishing_ids = prepare_master(args.master_path)
    units = unit_table(master, legitimate_ids, phishing_ids)

    # First primary split (iteration 1); its test side is set aside untouched.
    train_units, _ = grouped_stratified_split(units, args.test_size, args.seed)
    fit_units, val_units = grouped_stratified_split(train_units, args.validation_size, args.seed)

    def rows_for(unit_frame):
        legit = master["pair_id"].isin(unit_frame.loc[unit_frame["label_binary"].eq(0), "pair_id"]) & master["variant"].eq("legitimate")
        phish = master["pair_id"].isin(unit_frame.loc[unit_frame["label_binary"].eq(1), "pair_id"]) & master["variant"].eq("original_phishing")
        return master[legit | phish].reset_index(drop=True)

    fit, val = rows_for(fit_units), rows_for(val_units)
    vocab = build_vocab(fit["body_model"], args.max_vocab_size, args.min_freq)
    x_fit = encode_texts(fit["body_model"], vocab, args.max_length)
    y_fit = fit["label_binary"].to_numpy(dtype=np.int64)
    x_val = encode_texts(val["body_model"], vocab, args.max_length)
    y_val = val["label_binary"].to_numpy(dtype=np.int64)
    print(f"Device: {device}; fit {len(fit)} emails ({int(y_fit.sum())} phishing); "
          f"validation {len(val)} emails ({int(y_val.sum())} phishing)")

    rows = []
    for model_name in ["lstm", "gru", "cnn"]:
        set_all_seeds(args.seed)
        model = build_model(model_name, len(vocab), args).to(device)
        # Same loss, class weighting, optimizer and training step as RQ2 train_model().
        criterion = make_criterion(y_fit, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loader = make_loader(x_fit, y_fit, args.batch_size, shuffle=True)
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_one_epoch(model, loader, criterion, optimizer, args.grad_clip, device)
            val_loss, val_auc, val_dprime = evaluate(model, x_val, y_val, criterion, args.batch_size, device)
            rows.append({"model": model_name, "epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_auc": val_auc, "val_d_prime": val_dprime})
            print(f"{model_name} epoch {epoch:2d}: train_loss {train_loss:.4f}  "
                  f"val_loss {val_loss:.4f}  val_auc {val_auc:.4f}  val_d' {val_dprime:.3f}")

    curves = pd.DataFrame(rows)
    curves.to_csv(OUTPUT_DIR / "RQ2_epoch_check_curves.csv", index=False)

    verdict = []
    for model_name, c in curves.groupby("model", sort=False):
        best = c.loc[c["val_loss"].idxmin()]
        threshold = best["val_loss"] * (1 + TOLERANCE)
        chosen = c.loc[c["val_loss"].le(threshold)].iloc[0]
        # Training loss should fall overall; a rise between epochs flags instability.
        train_rises = int((c["train_loss"].diff() > 0).sum())
        verdict.append({"model": model_name, "best_epoch": int(best["epoch"]), "min_val_loss": best["val_loss"],
                        "within_5pct_threshold": threshold, "chosen_epochs": int(chosen["epoch"]),
                        "val_loss_at_chosen": chosen["val_loss"], "val_auc_at_chosen": chosen["val_auc"],
                        "epochs_where_train_loss_rose": train_rises})
    verdict = pd.DataFrame(verdict)
    verdict.to_csv(OUTPUT_DIR / "RQ2_epoch_check_verdict.csv", index=False)
    pd.DataFrame([
        {"item": "master_dataset_sha256", "value": sha256_file(args.master_path)},
        {"item": "split", "value": f"grouped by template_family_id; first primary split, seed {args.seed}, test_size {args.test_size}"},
        {"item": "validation", "value": f"{args.validation_size} of the training portion, whole template families"},
        {"item": "fit_emails", "value": len(fit)}, {"item": "validation_emails", "value": len(val)},
        {"item": "max_epochs", "value": args.max_epochs}, {"item": "grad_clip_max_norm", "value": args.grad_clip},
        {"item": "rnn_input", "value": "packed sequences (final state at last real token)"},
        {"item": "decision_rule", "value": "version 2: first epoch with val_loss within 5% of the model's minimum"},
        {"item": "device", "value": str(device)},
        {"item": "torch_version", "value": torch.__version__},
    ]).to_csv(OUTPUT_DIR / "RQ2_epoch_check_audit.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, (model_name, c) in zip(axes, curves.groupby("model", sort=False)):
            ax.plot(c["epoch"], c["train_loss"], marker="o", label="training loss")
            ax.plot(c["epoch"], c["val_loss"], marker="o", label="validation loss")
            chosen = int(verdict.loc[verdict["model"].eq(model_name), "chosen_epochs"].iloc[0])
            ax.axvline(chosen, color="grey", linestyle="--", label="chosen epochs")
            ax.set_title(f"{model_name.upper()} (chosen: {chosen})")
            ax.set_xlabel("epoch")
        axes[0].set_ylabel("weighted BCE loss")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "RQ2_epoch_check_curves.png", dpi=150)
    except ImportError:
        print("matplotlib not available; skipped the figure.")

    print("\n" + verdict.to_string(index=False))
    print("\nRQ2 arguments from this check: " + " ".join(
        f"--epochs-{m} {e}" for m, e in zip(verdict["model"], verdict["chosen_epochs"])))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
