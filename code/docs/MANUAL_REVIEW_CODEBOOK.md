# Manual Review Codebook: Source Emails

Use this codebook for the random samples in
`processed/dissertation_aligned_source_qc/manual_review_source_sample/`.
Each row is one email, shown as it is before any LLM enhancement.

## The question

**Is this a valid record of its expected class?**

- **Legitimate:** a standalone, human-written English email between people,
  sent in the course of work or personal correspondence.
- **Phishing:** a standalone English email that tries to deceive the reader
  into an action, such as clicking a link, opening a file, logging in,
  revealing credentials or personal data, or sending money.

Judge the email as it appears. Do not judge whether it is well written,
short, or contains typos: messy but valid emails are valid.

## How to fill in the columns

| Column | What to enter |
|---|---|
| `valid_record` | `yes` or `no` |
| `reason_code` | Required when `valid_record` is `no`: one code from the table below. Optional when `yes`: use `NOTE` to flag something worth mentioning. |
| `reviewer_notes` | Free text, a few words. Required for `OTHER` and `NOTE`. |

## Reason codes for `valid_record = no`

| Code | Meaning | Applies to |
|---|---|---|
| `WRONG_CLASS` | Not this class: a legitimate email that is really spam, marketing or phishing, or a phishing email that is really spam, marketing, a newsletter or an ordinary message | both |
| `AUTOMATED` | Sent by a system, newsletter or mailing list rather than a person | legitimate |
| `NON_ENGLISH` | Mainly another language, or contains a substantial non-English passage (translation or footer) | both |
| `NO_MESSAGE` | No real message: only a link, a signature or contact block, an attachment notice, or a "view this in HTML" notice | both |
| `FILLER` | The text is filler or word salad (e.g. novel excerpts or random phrases used to evade spam filters), not the actual message | both |
| `UNREADABLE` | Garbled or obfuscated so that it cannot be read as a message (split letters, interleaved characters, encoding debris throughout) | both |
| `NOT_LANGUAGE` | Mostly code, data or numbers, with no natural-language message | both |
| `NOT_STANDALONE` | Contains a forwarded message or reply chain from someone else | both |
| `OTHER` | Invalid for another reason; describe it in `reviewer_notes` | both |

## Borderline cases (decided in advance)

- **Tables, lists, rosters and shorthand** in legitimate business email are
  **valid** if a person wrote and sent them (decision D15).
- **A few garbled characters in a footer or quote**, in an otherwise readable
  email, are **valid** (decision D13); add `NOTE` if you want it recorded.
- **Obvious spelling and grammar errors** in phishing are **valid**: they are
  what the enhancement is meant to correct.
- **Templates imitating a company** (eBay, PayPal, banks) are **valid
  phishing** when they ask for an action.
- **Lottery, advance-fee and "you have won" scams** count as phishing when
  they ask the reader to reply, send money or provide personal data.
- **Pure advertising** (pharmacy, loans, products) with no deception or
  request for data is `WRONG_CLASS`.

## After the review

Run the summary (to be provided) to report, per class, the number of invalid
records and the 95% upper confidence bound on the invalid-record rate. With
0 invalid records in 100, the rate is below 3.0% with 95% confidence. Each
`no` decision is then confirmed and, if it holds, added to
`docs/manual_exclusions.csv`, after which the sweep and sample are re-checked.
