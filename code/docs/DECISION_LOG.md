# Decision Log

Each entry records a data decision, why it was made, and its measured effect.
Entries are added in order and never rewritten; a reversed decision gets a new
entry.

## D01 — 2026-10-06 — New working folder

- **Decision:** Rebuild the dataset in `data_quality_pipeline_v2/` from the raw
  files, leaving `pre_enhancement_source_qc_pipeline/` untouched as the prior
  reference.
- **Rationale:** Keep the prior dataset (SHA-256 `94b33c00…3e51d20`, used by the
  provisional RQ1 run) reproducible while fixes are applied.
- **Effect:** Raw files copied; checksums in `RAW_CHECKSUMS_SHA256.txt` match the
  source folder.

## D02 — 2026-10-06 — Date rule

- **Decision:** Any parseable date before 2020 is valid. Records with missing,
  unparseable, or future dates (2020 or later, e.g. erroneous 2038 or 2044
  dates) are excluded.
- **Rationale:** Pre-2020 limit reduces the chance of generative-AI-written
  emails; there is no lower bound.
- **Effect:** Matches existing code. The 1970-dated record (likely malformed
  header) is retained under this rule.

## D03 — 2026-10-06 — Deduplication and split policy

- **Decision:** Remove exact duplicates and near-duplicates (cosine similarity
  ≥ 0.95, character 5–7-gram TF-IDF) from the dataset. Assign template-family
  IDs at 0.80 similarity for both classes, for use in grouped train/test
  splits.
- **Rationale:** 0.95 removes copies; 0.80 families prevent template
  memorization across train and test.
- **Effect:** To be measured after rebuild. The split design is decided
  separately, after data quality is accepted.

## D04 — 2026-10-06 — Baseline reproduction from raw files

- **Decision:** Before any change, rerun the unmodified pipeline from the raw
  files in this folder.
- **Rationale:** Establish that the prior dataset is exactly reproducible, so
  every later difference is attributable to a logged decision.
- **Effect:** Exact reproduction. Enron 19,614 and Nazario queue 3,566 records;
  `Enron_SourceQC_Legitimate_Modeling_Text.csv` (`24cbad29…`) and
  `Nazario_LLM_Enhancement_Queue.csv` (`bc26e093…`) are byte-identical to the
  prior run. Log: `baseline_reproduction_run.log`.

## D05 — 2026-10-06 — Enhancement-fidelity QC

- **Decision:** An enhanced email is accepted only if it has a word ratio of
  0.70–1.30 relative to the original, keeps every original URL host and email
  address, and contains no model refusal or commentary. Prior-run outputs are
  reused only when the original text is byte-identical and the output passes
  this QC; other emails are re-attempted (same prompt, model and temperature,
  up to 4 attempts) and excluded if they still fail.
- **Rationale:** The prior QC checked only word count, which admitted 3
  refusals and 39 truncated rewrites.
- **Implementation note:** Links are compared by host, not full URL, because
  originals often contain URLs split by line wrapping that the model rejoins.
  A full-URL comparison flagged 128 prior outputs, mostly rejoined URLs; the
  host comparison flags 47, mostly real removals or originals with text glued
  to the address.
- **Effect on prior outputs:** 75 of 3,557 fail (42 too short, 3 refusals,
  33 link-only).

## D06 — 2026-10-06 — Shared final text normalization

- **Decision:** `body_model` for every row of every class is produced by one
  function (`scripts/text_quality.py: normalize_final_text`): zero-width
  characters removed; ©, ®, ™ removed; Unicode NFKC; typographic quotes,
  dashes and bullets mapped to ASCII; whitespace collapsed.
- **Rationale:** Enron is ASCII-only, while 26.7% of enhanced phishing emails
  contained typographic characters added by the LLM, which could act as a
  class or variant cue unrelated to fluency.
- **Effect (test build):** No row left the 25–500 word range. Residual
  non-ASCII share: legitimate 0.0%, original phishing 2.4%, enhanced
  phishing 1.5% (accented letters and look-alike characters that are part of
  the phishing content; retained and disclosed).

## D07 — 2026-10-06 — Post-enhancement exclusions

- **Decision:** Exclusions are applied to whole pairs, with the reason logged
  in `Final_Paired_Email_Dataset_Excluded_Rows.csv`:
  1. rows failing the final word-count/date screen after normalization;
  2. records in `docs/manual_exclusions.csv` (invalid source records missed by
     automated screens, keyed by raw source record);
  3. exact duplicate bodies within a variant after normalization, keeping the
     first pair.
- **Rationale:** The independent audit found (a) `phishing-2016#423`, whose
  body is only a software-license watermark, and (b) 11 enhanced outputs that
  duplicate another pair's output, from originals differing only in spacing or
  look-alike characters.
- **Effect (test build):** 12 pairs excluded (11 duplicates, 1 manual).
- **Related policy:** An enhanced output identical to its original is a valid
  result (the model judged no edit necessary) and is retained and reported.
  One case in the test build (`nazario_001453`, a fluent PayPal notice).

## D08 — 2026-10-06 — Template families and independent audit

- **Decision:** Each row carries `template_family_id` and
  `template_family_size` from connected components at cosine similarity ≥ 0.80
  (same representation as the 0.95 near-duplicate step), computed separately
  for legitimate emails and for original phishing emails; enhanced phishing
  inherits its pair's family. Each row also carries `source_record` (Enron
  maildir path, or Nazario file#index) for lineage.
- **Audit:** `scripts/audit_final_dataset.py` re-checks schema, pairs,
  inclusion, symmetry, uniqueness, enhancement fidelity, leakage-control
  columns and lineage on the final file alone, and exits non-zero on any
  failure.
- **Observation:** The largest legitimate family (291 emails) is one sender's
  mail sharing a long signature block, an author-identity cue that grouped
  splitting keeps out of the test set.

## D09 — 2026-10-06 — Full enhancement re-run (supersedes the reuse approach in D05)

- **Decision:** Re-enhance all 3,566 queued emails in one run instead of
  reusing 3,482 prior outputs. The model is pinned to the snapshot
  `gpt-5.4-mini-2026-03-17` (temperature 0.2, up to 4 attempts). The user
  prompt is unchanged: "Please correct any grammatical errors and improve the
  fluency of the following email without changing its intent." The system
  message adds a completeness instruction: "You edit email text only. Return
  only the revised email body. Return the complete email: keep every section,
  greeting, signature, footer, link, and email address from the original. Do
  not summarize, shorten, omit, or add content. If no corrections are needed,
  return the email unchanged." The D05 fidelity QC applies to every output.
- **Rationale:** One run under one QC regime and one model snapshot gives a
  single, simply described treatment. Under the prior system message the
  model systematically removed the eBay template from member-message lures, so
  re-attempts alone would have excluded those lures disproportionately.
- **Effect:** To be measured. The prior 100-pair manual review sample will be
  redrawn from the new outputs. The reuse option remains in the script but is
  not used.

## D10 — 2026-10-06 — Repeated text in originals (amends D09 and D05)

- **Decision:** (1) The system message allows removing accidental repetition:
  "You edit email text only. Return only the revised email body. Keep all of
  the original's content, including every section, greeting, signature,
  footer, link, and email address. Do not summarize or add content. Removing
  accidentally repeated words, sentences, or passages is part of the
  correction. If no corrections are needed, return the email unchanged."
  (2) The 0.70 lower word-ratio bound is computed against the original with
  repeated sentences (≥ 4 words, case-insensitive) counted once; the 1.30
  upper bound uses the full original. The independent audit implements the
  same rule separately.
- **Rationale:** Removing repeated words or passages is a legitimate grammar
  and fluency correction. 107 of 3,566 originals repeat at least one sentence
  (19 with ≥ 10% of words repeated, 6 with ≥ 30%), and 152 contain a doubled
  word. A "do not omit" instruction or a ratio against the full original
  would have penalized correct corrections (e.g. `nazario_000159`, whose body
  appears twice).
- **Effect on prior outputs:** Three correct de-duplications that failed the
  D05 rule now pass (`000159`, `000769`, `001925`); prior failures fall from
  75 to 72.
- **Pilot:** `Pilot_Enhancement_Pair_IDs.csv` = the 84 problem emails plus
  originals with ≥ 10% repeated words, run first with the final settings; its
  outputs are part of the full run.

## D11 — 2026-10-06 — Two-arm system-message pilot (supersedes the system messages in D09 and D10)

- **Decision:** The treatment is fluency and grammar correction only, so the
  instruction must not steer what the model keeps or omits. The D09/D10
  system messages listed email parts and repetition handling, which is
  guidance beyond that treatment, and are withdrawn. Two minimal arms are
  compared on the 99-email pilot, with the same user prompt, model snapshot,
  temperature, attempts and QC:
  - **A** (original run): "You edit email text only. Return only the revised
    email body."
  - **B** (A plus a scope sentence that restates the treatment): "You edit
    email text only. Correct grammar and improve fluency only; do not add,
    remove, or change any other content. Return only the revised email body."
- **Selection rule, fixed before the pilot is run:** Use B only if it
  improves the first-attempt QC pass rate without larger edits (median token
  edit distance and added-word share no higher than A's); otherwise use A.
  The chosen arm is used for the full run of all 3,566 emails, and the
  methods section quotes its system message and reports the pilot.
- **Unchanged from D10:** The lower word-ratio bound uses the original with
  repeated sentences counted once, so the QC accepts either keeping or
  removing accidental repetition and the instruction does not need to mention
  it.
- **Baseline:** The prior run's outputs for the 99 pilot emails pass the
  final QC for 18 of 99 and keep the eBay member-message template for 5 of 14.

## D12 — 2026-10-06 — Pilot result and decisions for the full run

- **Pilot result (99 emails, first model response per email):**

  | | Arm A | Arm B |
  |---|---|---|
  | Pass QC on first response | 27 | 77 |
  | Accepted after up to 4 attempts | 33 | 78 |
  | eBay member-message template kept | 8 / 14 | 14 / 14 |
  | Median token edit distance | 0.151 | 0.019 |
  | Median added-word share | 0.037 | 0.007 |

  Arm B made real but light edits: 5 / 99 outputs unchanged, 38 under 1%
  token change, 32 at 1–5%, 29 over 5%. Outputs:
  `processed/.../enhancement_pilot/` (`Pilot_Arm_Outcomes.csv`,
  `Pilot_Arm_Edit_Magnitude.csv`, `Pilot_Side_By_Side.csv`).
- **Decision 1, arm B for the full run:** B meets the D11 rule (higher pass
  rate, smaller edits). Treatment strength is reported via edit-distance
  distributions over the full run.
- **Decision 2, link-check measurement correction:** An original URL host or
  email address counts as kept if an output target extends it (truncated
  original), is contained in it (text glued to the original, e.g.
  `jose@monkey.orgaccount`), or equals it ignoring separators
  (`www.paypalcom`). Mentions in running text do not count. This corrects
  false failures caused by extraction artifacts, not the criterion: in arm B
  it cleared 6 of 12 link failures, and the remaining 6 are real changes
  (e.g. a misspelled scam domain "corrected", a recipient address replaced
  by an inferred first name). The audit implements the same rule separately.
- **Decision 3, word range:** The 25–500 word rule continues to apply to
  both originals and enhanced outputs (user decision).
- **Decision 4, manual exclusion:** `phishing2.mbox#1297` (`nazario_000646`)
  has a word-salad body and is added to `docs/manual_exclusions.csv`.
- **Full run:** All 3,566 queued emails in one fresh run with arm B, the
  pinned snapshot, temperature 0.2, up to 4 attempts, and the D05/D10/D12
  QC. Pilot outputs are not reused, so every enhanced email comes from the
  same run under the same QC.

## D13 — 2026-10-06 — Non-English, bilingual and garbled phishing emails

- **Finding:** The cleaning-stage language filter removes predominantly
  non-English emails (485 removed) but passes bilingual emails, because their
  English half supplies enough common English words, and very short foreign
  emails, which are too short for its ratio rules. In the 3,566-email queue:
  one entirely French email (`nazario_002836`), two English + French
  bilingual emails (`nazario_001736`, `nazario_002938`), and 7 English emails
  with a few garbled characters in footers or quotes (e.g. "Sociйtй en
  Commandite", "Copyright яFFFFA9").
- **Decision:** The French and bilingual emails are added to
  `docs/manual_exclusions.csv` so the corpus is English-only, as the methods
  section states, and French vocabulary does not occur only in the phishing
  class. The 7 emails with garbled boilerplate are kept and disclosed as a
  limitation. Re-cleaning them would change originals already being
  enhanced, and the garbling is a few boilerplate tokens.
- **Compatibility:** Manual exclusions are applied at the final build, so
  they do not affect the enhancement run in progress.

## D14 — 2026-10-06 — Enhancement paused until source data is final

- **Decision:** Stop the full arm-B run after 381 emails (380 accepted,
  1 rejected) and run enhancement once, only after the pre-enhancement data
  passes a systematic residual-error sweep and the resulting decisions are
  applied.
- **Rationale:** Any change to the original texts, or to the queue's row
  order (pair_ids are assigned by row order), would make outputs produced
  before the change stale. Enhancing once, on frozen inputs, keeps the
  treatment a single run.
- **Effect:** Partial outputs archived, not deleted, in
  `processed/dissertation_aligned_source_qc/superseded/partial_full_run_armB_stopped_2026-10-06/`.
  The next run starts from empty output files.

## D15 — 2026-10-06 — Residual-error sweep

- **Method:** `scripts/residual_error_sweep.py` re-screens every legitimate
  and phishing source record, independently of the original cleaning and
  source-validity screens. Review flags: foreign language (distinctive
  foreign function words, or mostly unknown words with almost no English
  function words), gibberish (≥ 30% of words neither in the bundled English
  word list `resources/english_wordlist_web2.txt` nor in ≥ 5 corpus emails),
  footer-only bodies, and phishing records with no link, address, account,
  payment, credential or file cue. Information flags (not errors): low-prose
  tables and lists, mostly numeric bodies, garbled boilerplate.
- **Result:** 102 records needed a decision (26 legitimate, 76 phishing).
  The reviewer excluded 66 and kept 36 as false flags. Decisions:
  `docs/residual_sweep_review_decisions.csv`; exclusions added to
  `docs/manual_exclusions.csv` (D15 rows).
  - Legitimate (13): non-English (6: Norwegian, Polish, French), English
    with a substantial non-English passage (1), promotional mailings (2), no
    message text (2: URL only, contact block only), SQL code (1), generated
    novelty text (1).
  - Phishing (53): filler text with no phishing message (42; image spam whose
    visible text is novel excerpts or generated word salad), English with a
    substantial non-English passage (3), spam rather than phishing (2), HTML
    notice only (1), mailbox system artifact (1), unreadable character
    obfuscation (1), readable character-level obfuscation (3; split letters
    or Greek look-alike letters, excluded because enhancing them would mainly
    measure de-obfuscation, not fluency).
- **Rule applied in both classes:** An email containing a substantial
  non-English passage, whether a translation or a footer, is excluded
  (extends D13).
- **Gaps in the original pipeline this exposes, to be stated in the methods:**
  the language filter was applied only to phishing, so the legitimate
  emails were never language-screened; and the filler-text screen missed 42
  image-spam records.
- **Information counts for disclosure:** 783 legitimate emails (4.0%) are
  mostly tables, lists or shorthand; 43 are mostly numeric.

## D16 — 2026-10-06 — Exclusions applied before enhancement; queue frozen

- **Decision:** Phishing manual exclusions are removed when the enhancement
  queue is built (`rebuild_strict_llm_enhancement_queue.py
  --manual-exclusions`), before pair_ids are assigned, so they are never
  enhanced and pair_ids have no gaps. Legitimate exclusions remain at the
  final build. The final build accepts phishing register entries only if
  they appear in the queue's `_Manual_Exclusions_Applied.csv`.
- **Rebuild from raw files:** Enron source file unchanged
  (`24cbad29…`); Nazario 0.95 modeling file 3,566 records; 58 manual
  exclusions applied (D07 1, D12 1, D13 3, D15 53).
- **Frozen enhancement queue:** `Nazario_LLM_Enhancement_Queue.csv`,
  3,508 records, SHA-256
  `2efc24cd5e1d0d26329378f5f1a551d89d39f1baf857997929921cdf012ffe5a`.
  A re-run of the sweep finds no records awaiting review.
- **Note:** The pilot files in `enhancement_pilot/` use the pair_ids of the
  earlier 3,566-record queue (`bc26e093…`).

## D17 — 2026-10-06 — Manual review of source emails before enhancement

- **Decision:** Before any enhancement, review simple random samples of the
  eligible source emails: 100 of 19,601 legitimate (after the 13
  legitimate manual exclusions) and 100 of the 3,508 queued phishing emails,
  seed 930 (`scripts/draw_source_review_sample.py`). Records are coded with
  `docs/MANUAL_REVIEW_CODEBOOK.md` (valid yes/no plus fixed reason codes;
  borderline cases defined in advance).
- **Rationale:** Measures the residual invalid-record rate with an exact 95%
  confidence interval (`scripts/summarize_source_review.py`), and catches any
  new type of problem before the treatment is applied, so enhancement runs
  once on confirmed source data.
- **Follow-up rule:** Each `no` is confirmed and added to the exclusion
  register. A new type of problem prompts a targeted search of the full
  data for that type before the queue is re-frozen.

## D18 — 2026-10-06 — Source review results and follow-up

- **Manual review result** (`Source_Review_Summary.csv`): legitimate 1 invalid
  in 100 (1.0%, exact 95% CI 0.03–5.4%); phishing 0 invalid in 100 (0.0%,
  exact two-sided 95% CI 0–3.6%; one-sided 95% upper bound 3.0%).
- **Follow-up, legitimate:** The invalid item (`legit_075`, a charity golf
  promotion sent to a distribution list) belongs to a type the original
  filters and the D15 sweep under-caught: external bulk and automated mail.
  A targeted search (sender outside enron.com plus commercial or event cues)
  returned 448 candidates, each read and decided with one rule. Exclude bulk
  or automated mail: marketing; conference, seminar or webcast promotion;
  newsletters and subscriptions; order, receipt, shipping, welcome and trial
  notices; organizational event mailings to member or student lists; job-
  posting lists; mass fundraising appeals; spam. Keep anything a person wrote
  to specific people: individually addressed invitations and pitches,
  meeting and conference-call logistics, working-group memos, speaker
  arrangements, personal mail. Result: 183 bulk/organizational and 24
  automated/transactional exclusions, 241 kept
  (`docs/enron_bulk_mail_review_decisions.csv`). With `legit_075`, 208 rows
  are added to `docs/manual_exclusions.csv`.
- **Relaxed-legitimate pool:** The 210 Enron emails excluded as promotional,
  bulk or automated mail (D15 and D18) are listed in
  `docs/relaxed_legitimate_pool.csv` as hard negatives for the planned
  sensitivity analysis.
- **Follow-up, phishing:** The reviewer noted an image placeholder ("[LOGO]").
  Such placeholders occur only in phishing (8 emails; Enron's are stripped in
  its cleaning step) and could mark the class. The shared final normalizer
  now removes image placeholders (`[LOGO]`, `[*.gif|jpg|png]`, `[cid:...]`) in
  every class, and the audit checks that none remain. This applies at the
  final build and does not change the frozen queue.
- **Effect on the frozen queue:** None. Legitimate exclusions apply at the
  final build; the queue SHA-256 remains `2efc24cd…`.

## D19 — 2026-10-07 — Full enhancement run, final build and audit

- **Enhancement:** One run of arm B over the frozen 3,508-email queue
  (`2efc24cd…`), model `gpt-5.4-mini-2026-03-17`, temperature 0.2.
  3,497 accepted (3,494 on the first attempt, 3 on the second), 11 excluded
  after 4 attempts (5 link or address removed, 4 too short, 2 word range),
  0 API errors. Log: `Nazario_LLM_Enhancement_Attempt_Log.csv`.
- **Final build:** 26,383 rows: 19,393 legitimate and 3,495 complete
  phishing pairs. Excluded at build: 221 legitimate (manual register) and 2
  phishing pairs whose enhanced text duplicated another pair's.
- **Final dataset SHA-256:**
  `dc0c00300bbe66698affe1f5eae2d4db59c738b9a035163aadbec40758665ecd`.
- **Independent audit:** PASS on every check
  (`Final_Dataset_Independent_Audit.csv`). Informational: 58 enhanced
  outputs (1.7%) are identical to their originals.
- **Treatment strength** (`Final_Enhancement_Edit_Magnitude.csv`, token-level
  edit distance between original and enhanced): unchanged at the token level
  373 (punctuation or case edits only, or none), under 1% 438, 1–5% 1,786,
  5–10% 675, 10–20% 208, 20% or more 15. Median 0.027, 90th percentile 0.086.
  Content retention median 0.977; added-word share median 0.015.
- **Residual non-ASCII share:** legitimate 0.0%, original phishing 2.3%,
  enhanced phishing 1.9% (accented letters and look-alike characters in
  phishing content).

## D20 — 2026-10-07 — Enhanced-pair review samples

- **Decision:** Review treatment fidelity on (a) a simple random sample of
  100 of the 3,495 final pairs (seed 930), which gives the unbiased failure
  rate, and (b) all 15 pairs with token edit distance ≥ 0.20, where edits
  beyond grammar and fluency are most likely (1 also falls in the random
  sample). Drawn with `scripts/enhancement_pair_review.py draw`; coded with
  `docs/ENHANCEMENT_REVIEW_CODEBOOK.md` (pass yes/no plus fixed reason
  codes; allowed edits and borderline cases defined in advance).
- **Follow-up rule:** Confirmed failures are excluded as whole pairs. If the
  random sample shows a failure type, a targeted search of all pairs for that
  type is run before the dataset is frozen.

## D21 — 2026-10-07 — Receiving-server stamps removed

- **Finding (reviewer):** Some phishing originals contain text inserted by
  the mail servers that received the corpus, not by the sender, e.g. a
  Spanish MailScanner footer ("Este mensaje ha sido analizado por
  MailScanner … y se considera que está limpio"). The most consequential
  variant is MailScanner's inline link warning, "MailScanner has detected a
  possible fraud attempt from … claiming to be …" (3 emails): a near-perfect
  phishing cue written by the recipient's server.
- **Rule:** Remove text inserted on the receiving side: MailScanner link
  warnings; MailScanner-format "scanned for viruses and dangerous content by
  <product>, and is believed to be clean" footers (MailScanner, ESVA,
  Pinpoint Securemail), English and Spanish, with the "MailScanner agradece
  a …" line; FortiGuard spam-submission controls. Keep sender-side text:
  antivirus signatures from the sender's own client (e.g. Avast), legal
  disclaimers, fake security lures (e.g. "upgrade to F-Secure HTK4S").
- **Implementation:** `SERVER_STAMP_RES` in `scripts/text_quality.py`,
  applied by the shared final normalizer to every row of every class. The
  arm-B model left the stamps verbatim, so originals and enhanced texts are
  cleaned identically and no re-enhancement is needed. The audit checks that
  none remain (`no_receiving_server_stamps`).
- **Effect:** 13 phishing pairs changed (both variants); no legitimate email
  matched. Two originals (`nazario_001182`, `nazario_001982`) became exact
  duplicates of other originals once their stamps were removed and were
  excluded by the existing duplicate rule. Neither is in the enhanced-pair
  review samples.
- **Final dataset (supersedes D19):** 26,379 rows: 19,393 legitimate and
  3,493 complete phishing pairs; SHA-256
  `45ddb140f202edb1d9424c9efabade9dd90a6d0b8f62ed489ddd0bc8343d56a0`.
  Independent audit: PASS.

## D22 — 2026-10-07 — Clean display text; repeated-passage pairs kept

- **Display column:** `body_display` in the final dataset now has the same
  collection artifacts removed as `body_model` (receiving-server stamps and
  image placeholders), via `clean_display_text` in `scripts/text_quality.py`.
  It keeps the original characters and line breaks for reading; the raw text
  remains traceable through `source_record`. Stamp patterns now match across
  line breaks. The audit checks both columns for stamps (PASS).
- **Final dataset (supersedes D21):** 26,379 rows: 19,393 legitimate and
  3,493 phishing pairs; SHA-256
  `b7994a8bff5557aef74831efeac9b11c9e64098b09d21d34ea94ba61b82971a3`.
  Independent audit: PASS.
- **Repeated-passage pairs, in response to methodologist feedback:** The
  methodologist judged that removing an entire repeated portion goes beyond
  grammar and fluency correction. In the final data this affects 2 of 3,493
  pairs: `nazario_001874` (251 → 119 words, of which 96 were in repeated
  sentences) and `nazario_000153` (308 → 152, body appears twice). All other
  pairs keep at least 74% of the original word count; the earlier
  291 → 85-word truncation pattern no longer occurs.
- **Decision (user):** Keep both pairs under D10. Justification: the
  removed text duplicates sentences already present in the email, so no
  information, request, link or persuasive element is lost; the remaining
  text matches the de-duplicated original (enhanced/de-duplicated ratios
  0.77 and 0.92), and removing accidental duplication is a standard
  fluency correction. These 2 pairs are reported, and a sensitivity analysis
  excluding them can be run at negligible cost.
- **Review samples redrawn:** The D20 samples had not been coded. They were
  archived (`superseded/enhanced_pair_review_pre_D22/`) and redrawn with the
  same seed and rules from the final 3,493-pair dataset, so the reviewer
  sees the cleaned text and the sample comes from the final population.

## D23 — 2026-10-07 — Look-alike letter obfuscation excluded; review files redrawn

- **Finding (reviewer, before coding began):** In `nazario_002723` the
  attacker replaced Latin "o" with Greek omicron throughout ("yοur
  accοunt", 42 words), and the enhancement converted every one back, which
  is de-obfuscation rather than grammar or fluency correction.
- **Targeted search (D20 follow-up rule):** Words mixing Latin with Greek or
  Cyrillic letters occur in 17 final phishing originals and in no legitimate
  email: 12 deliberate look-alike obfuscations (the enhancement removed it in
  9) and 5 encoding-damage cases ("Sociйtй", "MemberЎЁ", "CopyrightЉ"), which
  are kept under D13.
- **Decision (user):** Exclude the 12 obfuscated pairs, the same rule as
  D15 (readable character-level obfuscation), which the sweep missed
  because each word changes only one or two letters. Applied at the final
  build: `docs/manual_exclusions.csv` gains an `apply_at` column (`queue`
  for entries decided before enhancement, `final_build` for later ones), and
  the queue script uses only `queue` entries, so a rebuild from raw files
  still reproduces the frozen queue (`2efc24cd…`).
- **Final dataset (supersedes D22):** 26,355 rows: 19,393 legitimate and
  3,481 phishing pairs; SHA-256
  `3a7065aae4ca0bbf7146fcf92a947107f58c00d0f29aed89762bac5f58b96d5d`.
  Independent audit: PASS (informational: 7 rows contain mixed-script words,
  all encoding damage).
- **Review files:** Still uncoded, so archived
  (`superseded/enhanced_pair_review_pre_D23/`) and redrawn from the final
  population (100 random pairs, seed 930; 14 high-edit pairs). They are now
  written with a UTF-8 byte-order mark so Excel shows look-alike letters,
  accents and "→" correctly instead of decoding them as Mac Roman.

## D24 — 2026-10-07 — Enhanced-pair review coded; split-word obfuscation excluded; dataset frozen

- **Review result (reviewer, `docs/ENHANCEMENT_REVIEW_CODEBOOK.md`):** All
  114 coded pairs passed fidelity: random sample 0/100 failures (exact 95% CI
  0.000–0.036); high-edit census 0/14 failures. No reason codes were used.
- **Finding (reviewer note on `high_003`, `nazario_003487`):** The original
  breaks words apart ("Pl ease do not re ply to th is em ail"), and the
  enhancement rejoined them. The pair passed fidelity, because the meaning was
  unchanged, but the source is deliberate filter-evasion obfuscation, the
  class excluded under D15 and D23.
- **Targeted search (D20 follow-up rule):** Phishing originals and legitimate
  emails were scanned for adjacent fragments that join into a dictionary word
  (`resources/english_wordlist_web2.txt`) and for punctuation inserted inside
  words. 4 phishing originals are deliberate obfuscation:
  `nazario_003484` and `nazario_003487` (same WeTransfer template),
  `nazario_003001` (PayPal, "a ccount a ccess"; enhancement left it unchanged,
  edit 0.009) and `nazario_000759` ("E''nlarge"). Every legitimate hit was
  ordinary phrasing ("set up", "show up", "we're"). `nazario_000322` (4
  isolated line-wrap breaks such as "requir e") is kept.
- **Decision (user):** Exclude all 4 (`split_word_obfuscation`,
  `apply_at = final_build`). The rule is applied to the source email,
  regardless of what the enhancement did with it.
- **Review files are kept as coded.** No random-sample pair is affected. The
  3 high-edit items now excluded (`high_002`, `high_003`, `high_005`) stay
  in the file for the audit trail; `summarize` lists them under
  `excluded_after_review` and computes rates on pairs still in the dataset:
  random 0/100 (CI 0.000–0.036), high-edit 0/11 (CI 0.000–0.285).
  Coded-file SHA-256: random `9a41da0f…4c59`, high-edit `9c1759c6…1156`.
- **Final dataset — FROZEN (supersedes D23):** 26,347 rows: 19,393
  legitimate and 3,477 phishing pairs; template families at 0.80: 18,403
  legitimate, 2,560 phishing. SHA-256
  `25dcf2247880d249e5bd188c5c7624b843764838dc39c476543ad5ada86ac47d`.
  Independent audit: PASS. Any later change to the data requires a new
  decision-log entry and a new SHA.

## Pending

- Model training protocol and leakage prevention (grouped splits by template
  family), then RQ1–RQ3 analyses from the frozen dataset.
- Disclosure and sensitivity handling of the legitimate-class keyword filter
  (`docs/relaxed_legitimate_pool.csv`).
