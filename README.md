# Reproducibility Package

**Impact of Large Language Models on Phishing Email Detection: A Comparative Analysis of Classical Machine Learning and Deep Learning Models**
Guillaume Kanh, DSRT-930

This package contains the data, code, and outputs behind every result in Chapter 4.

**Prefer a notebook?** Open **`Reproducibility_Notebook.ipynb`** (Jupyter, VS Code, or Google Colab). It runs each option below in its own cell: view all results, check the files, a quick test rerun, and the full rerun.

**Want to see each step of the analysis?** Open **`Analysis_Walkthrough.ipynb`**. It follows Appendix E of the dissertation, from the dataset counts (Table C1) to Table 2, using the study's own code and saved outputs, and retrains the classical models for the first iteration to show that they reproduce the saved results. It runs in under a minute and needs no GPU.

## See the results without running anything

Open **`RESULTS_SUMMARY.html`** in a web browser: every Chapter 4 result (RQ1 to RQ3, Tables 1 and 2, the sensitivity analyses, AUC-ROC, and the outlier check), read directly from the output files.

To produce it yourself:

```
python print_results.py
```

This prints the same results in the terminal, confirms that the data files match their frozen checksums, and saves `RESULTS_SUMMARY.html`. Only the Python standard library is needed. On a Mac or Linux computer where `python` is not found, use `python3` in all commands.

## What is in the package

| Folder | Contents |
|---|---|
| `results/` | All outputs for RQ1 to RQ3: one folder per analysis, named as in Table 2, plus the outlier check and the Q-Q plots (Figures A3 to A5) |
| `review/` | Manual reviews: 100 random enhanced pairs, the most heavily edited pairs, 100 phishing and 100 legitimate source emails, the 102 flagged records, and the two review rubrics (codebooks) |
| `gap_search/` | Literature search of October 13, 2026: original database exports, deduplication log, and screening decisions. `About_This_Search.md` explains how each record was labeled by closeness to this study's design |
| `docs/` | `Decision_Summary.md` (data decisions in brief), `Sensitivity_Analyses_Explained.md`, and the full decision log |
| `data/raw/` | Raw Enron and Nazario files (the GitHub copy leaves out the 423 MB Enron archive; see below) |
| `data/final/` | Final dataset (`Final_Paired_Email_Dataset.csv`), the two sensitivity-analysis inputs, the independent audit of the final dataset, and the dataset preparation counts (Table C1) |
| `data/enhancement/` | The GPT-5.4-mini enhancement outputs and attempt log used to build the final dataset |
| `code/` | All scripts used by `run_all.py` |

## Rerun the study

```
pip install -r requirements.txt
python run_all.py --workspace <folder outside Dropbox or OneDrive>
```

`run_all.py` writes everything to the workspace folder and never changes the package. It:

1. Rebuilds the dataset from the raw files and stops if the enhancement queue or the final dataset differs from the frozen version. If the raw Enron archive is not in `data/raw/enron/` (it is too large for GitHub), the script first downloads the identical public file from https://www.cs.cmu.edu/~enron/ into the workspace and stops unless its checksum matches the study's copy.
2. Runs RQ1 on the CPU (about 1.5 hours for the primary analysis and 20 minutes per sensitivity analysis).
3. Runs RQ2 only if an NVIDIA GPU is available. Otherwise, run RQ2 on Google Colab with `RQ2_on_Colab.ipynb`; the notebook explains where to put the outputs.
4. Runs RQ3 and the outlier check.
5. Prints all rerun results and a check of each against the reported results ("match" or "differs"), and saves them as `RERUN_RESULTS_SUMMARY.html` in the workspace.

Other options: `--quick` (2 iterations per analysis, a fast test only), `--stage rq1` (one stage at a time), `--use-reported-rq2` (run RQ3 and the outlier check without a GPU, using the reported RQ2 outputs).

**The LLM enhancement is not rerun by default.** The rerun uses the saved enhancement outputs in `data/enhancement/`, so the rebuilt dataset matches the frozen one exactly. `--re-enhance` repeats the enhancement with your own `OPENAI_API_KEY`. It costs API credits, and because LLM output is not deterministic, the new dataset will differ from the frozen one.

## Notes

- **Result folder names follow Table 2.** Inside the files, the original run labels still appear: `grouped` (primary), `random`, `S2_masked` (masked identifiers), `S3_relaxed` (with bulk emails), `S4_no_repeated` (without repeated-passage pairs), and `S5` (outlier check).
- **Most heavily edited pairs.** 14 pairs with a token edit distance of .20 or more were reviewed, and none failed. 3 were later removed because the source email used split-word obfuscation, leaving the 11 reported in Chapter 3. The `status` column shows which.
- **Dataset counts (Table C1).** The rerun writes the count at each cleaning stage to the workspace (`processed/`). The counts from the study's rebuild are saved in `data/final/Dataset_Preparation_Counts.csv`.
- **CSV files** are saved as UTF-8 so Excel displays accented and look-alike characters correctly.
- **Data sources.** Enron Email Dataset (Cohen, 2015, https://www.cs.cmu.edu/~enron/) and Nazario Phishing Corpus (https://monkey.org/~jose/phishing/). Checksums of all files are in `CHECKSUMS_SHA256.txt`. In the GitHub copy, check them with `shasum -a 256 -c --ignore-missing CHECKSUMS_SHA256.txt`, because the Enron archive is listed there but not included.
