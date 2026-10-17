# Analysis plan: RQ1–RQ3

**Status: DRAFT. Not yet dated.**
Once the researcher dates and signs this plan (bottom of the page), the
analyses below are run once, exactly as written. Any later change is recorded
in `docs/ANALYSIS_DECISIONS.md` with its reason and reported as a deviation.

At the time of writing, no RQ1–RQ3 result has been produced on the frozen
dataset. The only model runs on it were the RQ2 epoch check (training data
only, A02) and code tests whose output was discarded unread.

## 1. Data

- `Final_Paired_Email_Dataset.csv`, frozen 2026-10-07 (DECISION_LOG D24),
  SHA-256 `25dcf2247880d249e5bd188c5c7624b843764838dc39c476543ad5ada86ac47d`.
- 19,393 legitimate emails (Enron) and 3,477 phishing pairs (Nazario
  original + LLM-enhanced), 26,347 rows. Models use the `body_model` column.
- Every script records the dataset SHA-256 in its audit file and the run is
  invalid if it differs.

## 2. Research questions and hypotheses (as in Chapter 3)

- **RQ1:** To what extent does LLM-enhancement of human-crafted phishing emails
  impact the discriminability of classical machine learning models?
  H01: no impact; Ha1: an impact.
- **RQ2:** The same for deep learning models. H02: no impact; Ha2: an impact.
- **RQ3:** How does the impact differ between classical machine learning and
  deep learning models? H03: no difference; Ha3: a difference.

## 3. Design (identical for RQ1 and RQ2)

- **Unit of sampling:** a legitimate email, or a phishing *pair* (original and
  its enhanced version always fall on the same side of a split).
- **Training set:** legitimate emails + **original** phishing emails only. The
  model never sees an enhanced email during training.
- **Two test sets per split, scored by the same trained model:**
  baseline = test legitimate + original phishing;
  enhanced = the same test legitimate + the enhanced versions of the same
  phishing pairs. The only difference between the two is the enhancement.
- **Splits:** test size 0.30, seed 930 + iteration index. **500 iterations
  for the primary analysis; 100 for each sensitivity run (S1–S4).** With the
  corrected resampled t-test the variance correction is dominated by the
  test/train ratio (1/500 + 0.4286 = 0.4306 vs 1/100 + 0.4286 = 0.4386), so
  100 iterations widen the confidence interval by under 1%. Because splits
  use seed 930 + iteration, the sensitivity runs use the first 100 splits of
  the corresponding primary or secondary design.
  - **Primary:** grouped by template family (`template_family_id`, 0.80
    character n-gram similarity). Each family is entirely in training or
    entirely in test, so near-copies of the same campaign or newsletter cannot
    appear on both sides. Families are assigned per class, so each split holds
    30% of each class (`rq_scripts/splits.py`).
  - **Secondary:** stratified random split by email or pair (the original
    design), reported for comparison with the grouped results.
- RQ1 and RQ2 use the same 500 splits for a given split type, so RQ3 compares
  the two families on identical data.

## 4. Models

**Classical (RQ1):** TF-IDF on word unigrams and bigrams (lowercase,
min_df 2, max 50,000 features), fitted on the training set of each split.

| Model | Settings |
|---|---|
| Logistic regression | liblinear, max_iter 1000, `class_weight='balanced'` |
| Linear SVM | `LinearSVC`, `class_weight='balanced'` |
| Random forest | 100 trees, `class_weight='balanced_subsample'` |

**Deep learning (RQ2):** vocabulary from the training set of each split
(regex tokens, max 20,000, min frequency 2), sequences truncated or padded to
300 tokens, embedding 100, dropout 0.30, Adam (lr 0.001), batch 128,
class-weighted binary cross-entropy (`pos_weight` = legitimate/phishing count
in the training set), gradient norm clipped at 1.0 (A01).

| Model | Architecture | Epochs (A02) |
|---|---|---|
| LSTM | 64 units, packed sequences (final state at last real token) | 5 |
| GRU | 64 units, packed sequences | 3 |
| CNN | 64 filters, kernel sizes 3, 4, 5, max pooling | 5 |

**Class imbalance:** about 5.6 legitimate emails per phishing email; every
model weights classes inversely to their frequency in the training set. No
emails are removed or duplicated.

No hyperparameter is tuned on test data. All settings above are fixed.

## 5. Measures

- **Primary: d′** (signal detection sensitivity) = z(hit rate) − z(false-alarm
  rate), phishing = signal. Computed from each model's predictions at its
  **default fixed cutoff** (probability 0.5; decision score 0 for the linear
  SVM; probability 0.5 for the deep learning models), the same cutoff for the
  baseline and enhanced test sets. Log-linear correction for extreme rates:
  (count + 0.5) / (n + 1) (Hautus, 1995).
- **Complementary: AUC-ROC**, which does not depend on a cutoff.
- **Per iteration:** each model's change = baseline − enhanced. The family
  change is the mean over the three models in that family. Positive values mean
  the enhancement reduced discriminability.

## 6. Statistical tests

- **RQ1, RQ2:** corrected resampled t-test (Nadeau & Bengio, 2003) on the 500
  family-level changes, H0: mean change = 0. Variance correction
  1/500 + 0.30/0.70 = 0.4306, df = 499, two-sided, α = .05.
- **RQ3:** the same test on the 500 per-iteration differences
  (classical change − deep learning change), H0: mean difference = 0.
- **Primary test per RQ:** d′ under the grouped split. Each RQ tests a
  separate, pre-stated hypothesis at α = .05. AUC-ROC, the random split and the
  sensitivity analyses support interpretation and are not used to accept or
  reject the hypotheses.
- **Reported for every test:** mean change with SD, t, df, p, and the 95%
  confidence interval of the mean change using the corrected variance, plus
  the per-model baseline and enhanced means.
- **Assumptions:** normality of the 500 differences assessed with a Q-Q plot
  (primary) and the Shapiro–Wilk test (complementary). With 500 observations
  the t-test is retained whatever these show; departures are reported and the
  outlier sensitivity analysis (S5) checks their influence.

## 7. Interpretation

- **Ceiling:** baseline d′ and AUC are reported first. If baseline AUC is near
  1, little room is left for a change, and this limits how large a drop the
  design can show. This is reported as a limitation, not as evidence either way.
- **Non-significant results** are reported as "no statistically significant
  change was detected", with the confidence interval showing the range of
  changes the data are compatible with. They are not described as proof of
  no effect.
- **Statistical vs practical significance:** with 500 splits, very small
  changes can be significant. Each significant result is interpreted with its
  size (mean change in d′ and AUC) and confidence interval.

## 8. Sensitivity analyses (all pre-specified, all reported)

| ID | Analysis | Purpose |
|---|---|---|
| S1 | Random split (section 3) | Effect of template-family leakage on the results |
| S2 | Corpus-identifying tokens masked in `body_model` before training and testing | Whether results rely on source artifacts rather than content |
| S3 | Relaxed legitimate set: 19,393 + the 210 bulk or automated Enron emails in `docs/relaxed_legitimate_pool.csv` (`sensitivity_S3/S3_Relaxed_Legitimate_Dataset.csv`, SHA-256 `2fdc429d…5b55`, A04) | Effect of the legitimate-class exclusions (harder negatives) |
| S4 | Exclude the 2 repeated-passage pairs kept under D22 | Influence of those pairs |
| S5 | Outlier rules on the 500 differences: 1.5 × IQR and ±3 SD, test re-run on the retained values | Influence of extreme iterations (as in the earlier outlier packet) |

S2–S4 use the grouped split. S5 uses the primary outputs and trains no new
models (`rq_scripts/run_s5_outlier_sensitivity.py`).

**S2 masking rule** (`rq_scripts/sensitivity.py`): the following are
**deleted** from `body_model` in every class and variant, identically:
email addresses, URLs, digit sequences, organisation and collection
identifiers (whole word, any case), and person names (whole word,
Capitalized or UPPERCASE only). Identifiers come from the reviewed candidate
list (`scripts/build_s2_mask_candidates.py`: words in ≥ 0.5% of one class's
emails and at most 5% as common in the other, each judged by what it is).
Person names are removed in both classes through a general first-name list
(`resources/first_names_propernames.txt`, macOS `propernames`) plus the
Enron names on the reviewed list; a general name is left out when the
corpus uses its lowercase form at least as often (an ordinary word, e.g.
"will"), and months and weekdays are left out. Deletion, rather than a
placeholder, is used because name presence itself differs by class (about
62% of legitimate vs 12% of phishing emails), largely because Enron is
internal corporate mail. Known limit: surnames are removed only when they
are Enron names on the list. The list is frozen as `docs/S2_Mask_Terms.csv`
(`scripts/freeze_s2_mask_terms.py`) **before** any RQ run.

**Not removable by masking (limitation):** all legitimate emails come from
one energy-trading company, so topic vocabulary (gas, power, deals) differs
from the phishing class. This is reported as a limitation of the corpora.

## 9. Execution

1. RQ1, RQ1 sensitivity runs and RQ3 on the local machine
   (`../.venv-source-qc`).
2. RQ2 on Google Colab (GPU), both split types and the sensitivity runs; the
   device and torch version are recorded in each audit file.
3. RQ3 after both families finish, per split type.
4. Results tables generated directly from the output CSV files.

## 10. Researcher sign-off

I approve this plan before any RQ1–RQ3 results on the frozen dataset exist.

Name: ______________________  Date: ______________
