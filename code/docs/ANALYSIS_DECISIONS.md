# Analysis decisions (model training and RQ1–RQ3)

Companion to `DECISION_LOG.md`, which covers the data (D01–D24). Every entry
here was made on the frozen dataset (SHA-256 `25dcf224…c47d`, D24) without
looking at RQ test-set results.

## A01 — 2026-10-07 — RQ2 recurrent models: packed sequences and gradient clipping

- **Finding (epoch check, run 1):** The first learning-curve check (rule: keep
  5 epochs only if validation loss at epoch 5 is within 5% of the minimum and
  the minimum is not before epoch 3) rejected 5 epochs for all three models.
  The curves also showed the LSTM and GRU were not training properly:
  training loss rose at epoch 8 and the LSTM then stayed flat at about 0.79
  (validation AUC near 0.5–0.9).
- **Causes, in `run_rq2_deep_learning.py`:**
  1. Texts are right-padded to 300 tokens and the LSTM/GRU classified from the
     hidden state after the whole sequence, i.e. after the trailing padding,
     so short emails were largely forgotten by the time of classification.
     The CNN (max pooling) was unaffected.
  2. No gradient clipping, so single large updates could derail recurrent
     training.
- **Fix (user approved):** The LSTM and GRU use packed sequences
  (`pack_padded_sequence`), so the final state is at each email's last real
  token (verified: identical output with and without padding). Gradient norm
  is clipped at 1.0 for all three models, so all are trained the same way.
  Layers, sizes, learning rate, batch size, class weighting and data are
  unchanged.
- **Consequence:** Earlier RQ2 results used the unfixed LSTM/GRU and are
  superseded by the next RQ2 run.

## A02 — 2026-10-07 — RQ2 epochs per model: LSTM 5, GRU 3, CNN 5

- **Procedure:** `rq_scripts/RQ2_deep_learning/scripts/check_rq2_epochs.py`,
  run on Google Colab (GPU) through `notebooks/RQ2_epoch_check_colab.ipynb`.
  Training portion of the first grouped split (template families, seed 930,
  test size 0.30); 15% of it held out as validation, again by whole template
  family (13,608 fit / 2,401 validation emails). Each model trained 15 epochs
  with the A01 fixes. The RQ2 test sets were never loaded.
- **Rule (version 2, fixed after A01 and before this run):** each model's
  epoch count is the first epoch whose validation loss is within 5% of that
  model's minimum over 15 epochs.
- **Result (run 2):**

  | Model | Minimum val. loss (epoch) | Chosen epochs | Val. loss at chosen | Val. AUC at chosen |
  |---|---|---|---|---|
  | LSTM | 0.0866 (5) | 5 | 0.0866 | 0.9970 |
  | GRU  | 0.1272 (4) | 3 | 0.1325 | 0.9941 |
  | CNN  | 0.0516 (11) | 5 | 0.0518 | 0.9990 |

  Training loss fell steadily for all models (only small fluctuations near
  zero). Beyond the chosen epochs, training loss keeps falling while
  validation loss stops improving, which marks the onset of overfitting.
- **Decision:** RQ2 defaults set to `--epochs-lstm 5 --epochs-gru 3
  --epochs-cnn 5`.
- **Limitation to report:** The curves come from one validation set; the 5%
  tolerance limits sensitivity to its noise.
- **Records:** Colab outputs `epoch_check/` (run 1) and `epoch_check_v2/`
  (run 2) to be copied from Google Drive into
  `rq_scripts/RQ2_deep_learning/output/`.

## A03 — 2026-10-07 — Splits, class weights, cutoff and confidence intervals implemented

- **Decisions (user):** d' at each model's default fixed cutoff; balanced
  class weights for all models; primary split grouped by template family,
  random split secondary.
- **Implementation:** `rq_scripts/splits.py` (`make_split`, `unit_table`;
  the grouped split raises an error if any family appears on both sides);
  `--split grouped|random` in RQ1, RQ2 and RQ3, with outputs in
  `output/<split>/`; `class_weight='balanced'` for logistic regression and
  the linear SVM; 95% confidence interval of the mean change (corrected
  variance) added to every corrected resampled t-test.
- **Verification:** all three scripts ran end to end for both split types
  (3 iterations, reduced settings). Only exit codes and audit fields (dataset
  SHA, split, class weights, cutoff, epochs) were inspected; the result files
  were deleted unread.
- **Context:** the earlier Chapter 4 RQ2 baseline (d' M = 3.03, SD = 1.05;
  AUC M = .88, SD = .11) is consistent with the unfixed LSTM/GRU in A01.

## A04 — 2026-10-07 — S3 relaxed legitimate dataset built and frozen

- **Build:** `scripts/build_s3_relaxed_legitimate_dataset.py`. The 210 Enron
  emails in `docs/relaxed_legitimate_pool.csv` (excluded at the final build
  only as bulk, promotional or automated mail; D15, D18) are taken unchanged
  from `Final_Paired_Email_Dataset_Excluded_Rows.csv`, so they carry exactly
  the final build's normalization and their original pair_ids. The script
  refuses to run unless the primary dataset matches the frozen SHA (D24).
- **Screens, as in the final build:** final quality screen, exact
  duplicates and near-duplicates (>= 0.95) against the primary legitimate
  emails and among themselves. All 210 passed every screen.
- **Template families:** recomputed at 0.80 over all 19,603 legitimate
  emails (18,595 families); phishing families unchanged. 31 added emails are
  in multi-member families (largest: 13 emails), and 4 families mix primary
  and added emails, which the grouped split keeps together.
- **Result:** `processed/dissertation_aligned_source_qc/sensitivity_S3/S3_Relaxed_Legitimate_Dataset.csv`,
  26,557 rows (19,603 legitimate, 3,477 phishing pairs), SHA-256
  `2fdc429d04576f2d41fa55a71f8049a6f92dd67fa9e9ad68b139bb7a13095b55`,
  reproduced byte-identically by a second build. Column
  `s3_relaxed_pool_member` marks the added emails. The primary dataset is
  unchanged. No model was trained.

## A05 — 2026-10-07 — Sensitivity runs implemented; S2 deletes names; 100 iterations

- **S2 (user decision after discussion):** identifiers and person names are
  deleted, not replaced by placeholders. The researcher noted that a
  `<NAME>` placeholder would remain a clue; with a general first-name list
  applied to both classes, names still appear in about 62% of legitimate vs
  12% of phishing emails. Names are removed in both classes through the
  general list plus the reviewed Enron names (`scripts/freeze_s2_mask_terms.py`,
  `rq_scripts/sensitivity.py`).
- **Options added to RQ1/RQ2:** `--mask-terms` (S2), `--exclude-pair-ids`
  (S4: `nazario_001874`, `nazario_000153`), `--run-label` (separate output
  folder per run); RQ3 takes `--run-label`; S5 in
  `rq_scripts/run_s5_outlier_sensitivity.py`. RQ2 on Colab through
  `notebooks/RQ2_all_runs_colab.ipynb`, with checkpoints saved to Google Drive.
- **Iterations (user decision):** primary 500; S1–S4 100 each (confidence
  interval under 1% wider; see plan section 3).
- **Verification:** code tests on 2–3 iterations; only exit codes and audit
  fields were inspected, outputs deleted unread.

## A06 — 2026-10-07 — S2 mask list reviewed and frozen

- **Review (researcher):** delete all 25 organisation/collection identifiers
  (Enron name, offices, units, internal systems; `jose`, `xxxx`); keep the
  borderline general words (corp, ext, sr, doc, pl, dg, pg, austin,
  northern, southern, canadian); keep phishing-side brand details copied from
  impersonated companies (usaa, hamilton, fredericksburg, pp, adme, ncrcp);
  all other candidates as suggested. Decisions and notes are recorded in
  `S2_Mask_Candidates_Review.csv`.
- **Frozen:** `docs/S2_Mask_Terms.csv` (25 org identifiers, 1,263 person
  names), SHA-256
  `bf30e114399f013ea91256756556a7804185006d5de5c82b0bb2019b6af87e68`.
  Name list source `resources/first_names_propernames.txt`, SHA-256
  `626d634b40b1ad9257d0e4f16e155ea87d258dd458e9ccf94fb84bb1b63e585a`.
- The RQ2 Colab notebook checks all three frozen fingerprints (primary, S3,
  S2) before running.

## A07 — 2026-10-07 — RQ2 runtime logging added after the primary run

- **Context:** RQ1 records its runtime (primary: 5,971.7 s for 500
  iterations, about 11.9 s per iteration, local Mac CPU). The RQ2 script
  never did, so the completed RQ2 primary run (grouped, 500/500 iterations,
  GPU on Colab) has no recorded runtime.
- **Change (timing only, no effect on models, splits or results):** RQ2 now
  stores wall-clock seconds per model fit (`model_seconds`) and per
  iteration (`iteration_seconds`) in the iteration results, and reports
  timed iterations, total compute seconds and mean seconds per iteration and
  per model fit in the audit. Totals add up correctly across resumed
  sessions. Applies to the four RQ2 sensitivity runs.
- **Reporting:** compute time for deep learning is reported from the
  sensitivity runs (seconds per iteration, Colab GPU), with the hardware
  stated, next to RQ1's (local CPU).

## A08 — 2026-10-08 — All pre-specified analyses completed

- **Runs (all verified: frozen inputs, requested = completed iterations,
  no missing or duplicate iteration/model rows):** RQ1 (local CPU) and RQ2
  (Colab GPU) for grouped (500), random, S2_masked, S3_relaxed and
  S4_no_repeated (100 each); RQ3 for each, pairing the same-label RQ1/RQ2
  outputs; S5 on the grouped outputs. Outputs in
  `rq_scripts/RQ{1,2,3}_*/output/<run>/` and
  `rq_scripts/S5_outlier_sensitivity/output/grouped/`.
- **Primary results (d', corrected resampled t-test, df = 499):**
  RQ1 change 0.004, 95% CI −0.015 to 0.024, p = .662;
  RQ2 change 0.012, 95% CI −0.015 to 0.038, p = .383;
  RQ3 difference −0.007, 95% CI −0.038 to 0.023, p = .633.
  H01, H02 and H03 not rejected. AUC results agree.
- **Sensitivity analyses S1–S5:** same conclusions in every analysis (all
  p > .16); outlier rules change no p-value materially.
- **Not as planned:** the RQ2 sensitivity runs used the script version
  without the A07 timing code (the updated file was not uploaded), so no
  RQ2 runtime was recorded; results are unaffected because that code only
  added timing.

## Pending

- Optional: short timing-only run on Colab (RQ2, and RQ1 on the same
  hardware) for the compute-time comparison.
- Reproducibility clean-up once all analyses are done (researcher's
  decision: nothing is moved before then), including: 88 Dropbox
  "conflicted copy" files of `Nazario_LLM_Enhanced_Phishing_Emails.csv` in
  `processed/dissertation_aligned_source_qc/` (Oct 6 enhancement run; the
  pipeline reads only the exact file name, so the frozen dataset is
  unaffected); duplicate copies of the RQ scripts in older project folders;
  `colab_upload/` staging folder; complete record-flow table with the final
  manual-review and build stages; one run-everything README for the
  methodologist package.
