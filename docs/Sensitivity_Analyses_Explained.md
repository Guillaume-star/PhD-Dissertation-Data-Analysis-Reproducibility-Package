# What Each Sensitivity Analysis Tests

Each sensitivity analysis changes one decision that could reasonably be questioned
and checks whether the result depends on it. Each repeats RQ1 to RQ3 with 100
train-test splits. Results are in Table 2 and in `results/`.

**Result:** in all four analyses, no change in d′ was statistically significant for
either model family, and the two families did not differ (all p > .16).

| Analysis | The question it answers | What was changed |
|---|---|---|
| **Random split** | "The grouped split is unusually strict. Would the usual random split give a different result?" | Emails were split at random instead of keeping template families together. Baseline d′ was higher, as expected when similar emails appear in both training and testing, but the effect of enhancement stayed non-significant. |
| **Masked identifiers** | "The two classes come from two different collections. Are the models telling Enron from Nazario instead of legitimate from phishing?" | Email addresses, URLs, numbers, terms that identify either collection, and person names were deleted from every email. |
| **With bulk emails** | "Removing newsletters, marketing, and automated notifications made the legitimate class too easy to recognize." | The 210 bulk, promotional, and automated Enron emails that had been excluded were added back to the legitimate class. These are the legitimate emails that most resemble phishing. |
| **Without repeated-passage pairs** | "Some enhancements went beyond grammar and fluency correction." | The two pairs in which the model removed a whole repeated passage were dropped. |

**Outlier check.** The RQ2 and RQ3 difference scores departed from normality. The
corrected resampled t-tests were repeated after removing extreme scores (1.5 × IQR
rule and ±3 SD rule). The conclusions did not change.
