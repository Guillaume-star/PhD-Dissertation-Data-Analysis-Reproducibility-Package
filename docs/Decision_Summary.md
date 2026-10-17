# Data Decisions in Brief

The main decisions that shaped the dataset, and why. The entry number in brackets
points to the full record in `Decision_Log_Full.md`.

## Selecting emails

1. **Two public sources.** Legitimate emails come from the Enron Email Dataset, phishing emails from the Nazario Phishing Corpus.
2. **Emails dated before 2020 only.** Keeps "legacy" emails that were written before modern generative AI tools were widely available. [D02]
3. **Standalone emails of 25 to 500 words.** Removes replies, forwards, and messages too short or too long to be a normal single email.
4. **Exact and near-duplicates removed** (cosine similarity of .95 or higher). Prevents copies of the same email from being counted more than once. [D03]

## Removing invalid records

5. **Non-English, gibberish, filler, and obfuscated emails removed.** A new screening step flagged foreign-language text, gibberish, phishing records with no phishing cue (no link, account, payment, or credential request), and emails with split or look-alike letters. Every flagged record was read by a person before exclusion: 102 flagged, 66 excluded, 36 kept. Most exclusions were image spam whose visible text was novel excerpts or word salad. [D13, D15, D23, D24]
6. **Bulk, promotional, and automated Enron emails removed** (newsletters, marketing, notifications). They are not person-to-person email and could act as an easy clue for "legitimate." The "with bulk emails" sensitivity analysis adds them back. [D15, D18]
7. **Text added by the receiving mail server removed** (for example, "MailScanner has detected a possible fraud attempt"). It was added by the recipient's server, not the sender, and would reveal the label. [D21]
8. **Typographic characters standardized in every email** (curly quotes, dashes, symbols). The LLM added these characters, so they could have marked enhanced emails. [D06]
9. **Random check of the cleaned data before enhancement.** 100 phishing emails: 0 invalid. 100 legitimate emails: 1 invalid, which led to the bulk-mail removal in item 6. [D17, D18]

## Enhancing the phishing emails

10. **One model, one prompt, one run.** GPT-5.4-mini (snapshot 2026-03-17), temperature 0.2, the same user prompt for every email. [D09, D19]
11. **System message limits the edit to grammar and fluency.** Chosen in a 99-email pilot because it produced smaller edits and fewer failures than a shorter system message (median change of about 2% of words, against about 15%). [D11, D12]
12. **Automatic quality check on every enhanced email.** Accepted only if it kept 70% to 130% of the original word count (repeated sentences counted once for the lower bound), kept every link and email address, and contained no refusal or comment from the model. Failures were retried up to 4 times; 11 emails still failed and were excluded. [D05, D10, D19]
13. **Manual review of enhanced pairs.** 100 random pairs and every heavily edited pair (token edit distance of .20 or more) were coded with a written rubric. No pair failed. [D20, D24]
14. **Two pairs where the model removed a repeated passage were kept,** because the removed text repeated sentences already in the email. The "without repeated-passage pairs" sensitivity analysis drops them. [D22]

## Preparing the analysis

15. **Template families for the train-test splits.** Emails with similarity of .80 or higher form a family, and a family is never split between training and testing. This prevents the models from memorizing a template. [D03, D08]
16. **Dataset frozen before any analysis.** Final dataset: 19,393 legitimate emails and 3,477 phishing pairs (original and enhanced). Its SHA-256 checksum is recorded, so any change to the data would be detectable. [D24]
