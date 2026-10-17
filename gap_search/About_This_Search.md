# About the Targeted Literature Search

This search was run to check the research gap stated in Chapter 2 (Databases and Search Terms): whether any prior study had tested how subtle LLM enhancement affects phishing detection using this study's design.

## The search

- **Date:** October 13, 2026
- **Databases:** Scopus, IEEE Xplore, ProQuest, and EBSCOhost eBook Collection
- **Years:** 2022 to 2026
- **Search terms:** four groups of terms combined with AND: (1) phishing or spam, (2) large language models, (3) modification (rephrasing, rewriting, paraphrasing, enhancement, grammar, fluency), and (4) detection or classification. The exact string for each database is in Table A5 of the dissertation.

## From 568 records to the result

| Step | Records | File |
|---|---|---|
| Records exported from the databases | 568 (Scopus 318, IEEE Xplore 144, ProQuest 106, EBSCOhost 0) | `Search_Export_*` |
| Duplicates removed | 155 | `Deduplication_Log.csv` |
| Records screened by title and abstract | 413 | `Gap_Search_Screening_Decisions.csv` |
| Relevant: tested a detector on LLM-generated or LLM-modified email | 30 (12 close to this design, 18 not close) | same file |
| Matched the full design of this study | 0 | |

**Duplicates.** Records with the same DOI or the same title were treated as one study. The published version was kept when a study also appeared as a preprint.

## How each record was labeled

Each of the 413 records has one label in the column `closeness_to_this_design`:

| Label | Records | Meaning |
|---|---|---|
| **Close to this design** | 12 | Tested a detector on LLM-generated or LLM-modified email **and** shares at least one element of this study's design: modification of existing emails, a before-and-after comparison, a discriminability measure, or an inferential test |
| **Relevant, not close** | 18 | Tested a detector on LLM-generated or LLM-modified email, but shares none of those design elements. Most propose a new detector and test it on fully LLM-generated emails |
| **Not relevant** | 383 | Did not test a detector on LLM-generated or LLM-modified email |

**This study's full design** combines all four elements: subtle enhancement of existing emails (grammar and fluency only), a paired before-and-after comparison of the same emails, a discriminability measure (d′), and inferential statistical testing. None of the 413 records combined them. The closest studies modified emails more extensively or with the aim of evading detection, and reported accuracy or evasion rates.

**Common reasons a record was not relevant** (the `reason` column gives the specific reason for each record):

- **No detection model was tested.** For example, a study analyzed the language of AI-generated phishing, or measured how users respond to it.
- **The LLM-generated emails were used only for training.** The detector was never evaluated on them.
- **The emails were not LLM-generated or LLM-modified.** For example, LLMs were used as detectors on ordinary phishing.
- **Not email.** For example, phishing websites, URLs, SMS, or chat messages.
- **Not a study.** For example, a literature review without its own evaluation, or conference front matter.
- **Off topic.** For example, general machine-generated text detection, malware, or other security topics.

## Columns in the screening file

| Column | Meaning |
|---|---|
| `closeness_to_this_design` | Close to this design; Relevant, not close; or Not relevant, as defined above |
| `reason` | What the study did, and why it received its label |
| `cited_in_dissertation` | yes when the study is cited in the dissertation |
| other columns | Bibliographic details and abstract, as exported from the databases |
