# Manual Review Codebook: Enhanced Phishing Pairs

Use this codebook for the files in
`processed/dissertation_aligned_source_qc/manual_review_enhanced_pairs/`.
Each row is one phishing email shown twice: the original and its
LLM-enhanced version.

## The question

**Did the enhancement only correct grammar and improve fluency, leaving
everything else as it was?**

The treatment in this study is a *subtle* enhancement. The enhanced email
should send the same message, to the same reader, with the same request,
links, and attack strategy, only written more correctly and more fluently.

## How to read a row

| Column | What it shows |
|---|---|
| `word_changes` | Every word-level edit as `[old → new]`. `∅` means nothing (a pure insertion or deletion). Start here. |
| `original_body`, `enhanced_body` | The two full texts, to check context. |
| `token_edit_distance` | Share of words changed: 0 = identical wording, 0.05 = about 5%. |
| `content_retention` | Share of the original's distinct words still present. |
| `added_word_share` | Share of the enhanced text's words that never appear in the original. |

## How to fill in the columns

| Column | What to enter |
|---|---|
| `fidelity_pass` | `yes` if every change is grammar, spelling, punctuation, word order or fluency; `no` otherwise |
| `reason_code` | Required when `fidelity_pass` is `no`: one code from the table below. Optional when `yes`: `NOTE` to record something. |
| `reviewer_notes` | A few words. Required for `OTHER` and `NOTE`. |

## Reason codes for `fidelity_pass = no`

| Code | The enhancement … |
|---|---|
| `MEANING_CHANGED` | changed what the email says: a fact, amount, date, deadline, consequence or request differs |
| `CONTENT_REMOVED` | dropped a sentence, section, greeting, signature or footer that carried content |
| `CONTENT_ADDED` | added information, sentences or details not in the original |
| `PERSUASION_ADDED` | made the attack more convincing beyond fluency: more urgency, threat, authority, reassurance or personalization |
| `LINK_OR_ADDRESS_CHANGED` | altered a URL, domain, email address, phone number or account number, including "fixing" a misspelled scam domain |
| `STRUCTURE_CHANGED` | reorganized the email: moved, merged or split sections so the layout or flow of the lure differs |
| `DEOBFUSCATED` | undid deliberate obfuscation (spaced-out letters, look-alike characters) rather than correcting genuine errors |
| `OTHER` | another change beyond grammar and fluency; describe it |

## Decided in advance

- **Allowed (still `yes`):** fixing spelling, grammar, agreement, tense,
  punctuation, capitalization and spacing; smoothing awkward phrasing with the
  same meaning; removing accidentally repeated words or passages; repairing
  URLs split by line wrapping.
- **No change at all** (the model left the email as it was) is `yes`: the
  model judged no correction necessary. Add `NOTE` if the original clearly
  had errors that were not fixed.
- **Small tone shifts** that come from correct grammar alone (e.g. "pls
  click" → "please click") are `yes`. Code `PERSUASION_ADDED` only when the
  edit adds pressure, threat, authority or personalization that was not there.
- **Expanding an abbreviation** or **correcting a company name's spelling**
  (e.g. "Paypall" → "PayPal") in running text is `yes`; changing a domain or
  address is `LINK_OR_ADDRESS_CHANGED`.

## After the review

`scripts/enhancement_pair_review.py summarize` reports the fidelity-failure
rate per file with an exact 95% confidence interval. The random file gives the
unbiased rate for the whole treatment; the high-edit file checks the pairs
where a failure is most likely. Failures are then confirmed, and the affected
pairs are excluded before the dataset is frozen.
