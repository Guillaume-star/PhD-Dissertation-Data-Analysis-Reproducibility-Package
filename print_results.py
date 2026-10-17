"""Print all Chapter 4 results from the output files, and save them as a document.

    python print_results.py                      reported results -> printed and saved as RESULTS_SUMMARY.md
    python print_results.py --compare WORKSPACE  results of a rerun made with run_all.py, plus a check
                                                 against the reported results -> saved in WORKSPACE

Every number is read from the CSV output files; nothing is typed in by hand.
Only the Python standard library is needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
RESULTS = PACKAGE / "results"

ANALYSES = [  # (folder name, label in Table 2)
    ("primary_grouped_split", "Primary (grouped split)"),
    ("random_split", "Random split"),
    ("masked_identifiers", "Masked identifiers"),
    ("with_bulk_emails", "With bulk emails"),
    ("without_repeated_passage_pairs", "Without repeated-passage pairs"),
]
RQ_DIRS = {"RQ1": "RQ1_classical_ml", "RQ2": "RQ2_deep_learning", "RQ3": "RQ3_model_family_comparison"}
MODEL_NAMES = {"logistic_regression": "Logistic Regression", "linear_svm": "Linear SVM", "random_forest": "Random Forest",
               "cnn": "CNN", "gru": "GRU", "lstm": "LSTM"}
DATA_SHA256 = {
    "data/final/Final_Paired_Email_Dataset.csv": "25dcf2247880d249e5bd188c5c7624b843764838dc39c476543ad5ada86ac47d",
    "data/final/With_Bulk_Emails_Dataset.csv": "2fdc429d04576f2d41fa55a71f8049a6f92dd67fa9e9ad68b139bb7a13095b55",
    "data/final/Masked_Identifier_Terms.csv": "bf30e114399f013ea91256756556a7804185006d5de5c82b0bb2019b6af87e68",
}
TOLERANCE = 1e-6


class Source:
    """Where a set of outputs lives: the reported results/ folder, or a rerun workspace."""

    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace

    def folder(self, rq: str, analysis: str) -> Path:
        if self.workspace is None:
            return RESULTS / RQ_DIRS[rq] / analysis
        return self.workspace / "rq_scripts" / RQ_DIRS[rq] / "output" / analysis

    def outlier_file(self) -> Path:
        if self.workspace is None:
            return RESULTS / "outlier_check" / "S5_outlier_sensitivity_summary.csv"
        return self.workspace / "rq_scripts" / "S5_outlier_sensitivity" / "output" / "primary_grouped_split" / "S5_outlier_sensitivity_summary.csv"

    def rows(self, rq: str, analysis: str, name: str) -> list[dict]:
        return read(self.folder(rq, analysis) / f"{rq}_{name}.csv")

    def ttest(self, rq: str, analysis: str, metric: str) -> dict | None:
        return next((r for r in self.rows(rq, analysis, "corrected_resampled_t_tests") if r["metric"] == metric), None)


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def f(value, decimals: int = 3) -> str:
    return f"{float(value):.{decimals}f}"


def apa(value, decimals: int = 3) -> str:  # .999 instead of 0.999, for values bounded by 1
    return f(value, decimals).replace("0.", ".", 1)


def p_text(p) -> str:
    p = float(p)
    return "p < .001" if p < 0.001 else "p = " + apa(p)


def t_text(row: dict) -> str:
    return f"t({row['degrees_of_freedom']}) = {float(row['t_statistic']):.2f}, {p_text(row['p_value_two_sided'])}"


def change_text(row: dict | None, decimals: int = 3) -> str:
    if row is None:
        return "not run"
    m, lo, hi = (f(row[k], decimals) for k in ("mean_difference", "ci95_low", "ci95_high"))
    return f"{m} [{lo}, {hi}], {p_text(row['p_value_two_sided'])}"


def shapiro(src: Source, rq: str, analysis: str, metric: str = "d_prime") -> str:
    row = next((r for r in src.rows(rq, analysis, "normality_assumption_summary") if r["metric"] == metric), None)
    return "not available" if row is None else p_text(row["shapiro_wilk_p_value"])


def family(src: Source, rq: str, analysis: str, metric: str) -> dict | None:
    return next((r for r in src.rows(rq, analysis, "metric_summary") if r["group"] == "family" and r["metric"] == metric), None)


def rq12_section(src: Source, rq: str, title: str) -> list[str]:
    a = "primary_grouped_split"
    d, auc = family(src, rq, a, "d_prime"), family(src, rq, a, "roc_auc")
    t_d, t_auc = src.ttest(rq, a, "d_prime"), src.ttest(rq, a, "roc_auc")
    if not (d and t_d):
        return [f"## {title}", "", "Not run yet.", ""]
    out = [f"## {title}", "",
           f"- Normality of the {t_d['n_observations']} paired difference scores: Shapiro-Wilk {shapiro(src, rq, a)}",
           f"- Baseline d′: M = {f(d['baseline_mean'])}, SD = {f(d['baseline_sd'])}",
           f"- Post-enhancement d′: M = {f(d['enhanced_mean'])}, SD = {f(d['enhanced_sd'])}",
           f"- Mean change in d′ (baseline − enhanced): {f(t_d['mean_difference'])}, 95% CI [{f(t_d['ci95_low'])}, {f(t_d['ci95_high'])}], {t_text(t_d)}"]
    if auc and t_auc:
        out += [f"- Baseline AUC-ROC: M = {apa(auc['baseline_mean'])}, SD = {apa(auc['baseline_sd'])}",
                f"- Post-enhancement AUC-ROC: M = {apa(auc['enhanced_mean'])}, SD = {apa(auc['enhanced_sd'])}",
                f"- Mean change in AUC-ROC: {f(t_auc['mean_difference'], 5)}, {t_text(t_auc)}"]
    return out + [""]


def rq3_section(src: Source) -> list[str]:
    a = "primary_grouped_split"
    s = next((r for r in src.rows("RQ3", a, "family_change_summary") if r["metric"] == "d_prime"), None)
    t_d, t_auc = src.ttest("RQ3", a, "d_prime"), src.ttest("RQ3", a, "roc_auc")
    if not (s and t_d):
        return ["## RQ3: Difference between model families (primary analysis)", "", "Not run yet.", ""]
    out = ["## RQ3: Difference between model families (primary analysis)", "",
           f"- Normality of the {t_d['n_observations']} paired differences: Shapiro-Wilk {shapiro(src, 'RQ3', a)}",
           f"- Change in d′, classical ML: M = {f(s['classical_mean_change_baseline_minus_enhanced'])}, SD = {f(s['classical_sd'])}",
           f"- Change in d′, deep learning: M = {f(s['deep_learning_mean_change_baseline_minus_enhanced'])}, SD = {f(s['deep_learning_sd'])}",
           f"- Mean difference (classical − deep learning): {f(t_d['mean_difference'])}, 95% CI [{f(t_d['ci95_low'])}, {f(t_d['ci95_high'])}], {t_text(t_d)}"]
    if t_auc:
        out.append(f"- Mean difference in AUC-ROC change: {f(t_auc['mean_difference'], 5)}, {t_text(t_auc)}")
    return out + [""]


def table1(src: Source) -> list[str]:
    out = ["## Table 1: Model-level average d′ (primary analysis; descriptive only)", "",
           "| Model family | Model | Baseline mean d′ | Enhanced mean d′ | Mean difference |", "|---|---|---|---|---|"]
    for rq, fam in (("RQ1", "Classical ML"), ("RQ2", "Deep learning")):
        for r in sorted((r for r in src.rows(rq, "primary_grouped_split", "metric_summary")
                         if r["metric"] == "d_prime" and r["group"] != "family"), key=lambda r: MODEL_NAMES.get(r["group"], r["group"])):
            out.append(f"| {fam} | {MODEL_NAMES.get(r['group'], r['group'])} | {f(r['baseline_mean'])} | {f(r['enhanced_mean'])} | "
                       f"{f(r['mean_difference_baseline_minus_enhanced'])} |")
    return out + [""]


def table2(src: Source, metric: str, title: str, decimals: int) -> list[str]:
    out = [f"## {title}", "", "| Analysis | Iterations | RQ1: Classical ML | RQ2: Deep learning | RQ3: Classical − DL |", "|---|---|---|---|---|"]
    for folder, label in ANALYSES:
        rows = [src.ttest(rq, folder, metric) for rq in RQ_DIRS]
        n = next((r["n_observations"] for r in rows if r), "")
        out.append(f"| {label} | {n} | " + " | ".join(change_text(r, decimals) for r in rows) + " |")
    return out + ["", "Values are the mean change (baseline − enhanced), 95% CI, and p from corrected resampled t-tests.", ""]


def outlier_section(src: Source) -> list[str]:
    rules = {"none": "No removal", "1.5_IQR": "1.5 × IQR rule", "3_SD": "±3 SD rule"}
    rows = [r for r in read(src.outlier_file()) if r["metric"] == "d_prime"]
    out = ["## Outlier check (primary analysis, d′)", ""]
    if not rows:
        return out + ["Not run yet.", ""]
    out += ["| RQ | Rule | Scores removed | Mean change [95% CI] | Test |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['rq']} | {rules.get(r['rule'], r['rule'])} | {r['n_removed']} | "
                   f"{f(r['mean_difference'])} [{f(r['ci95_low'])}, {f(r['ci95_high'])}] | {t_text(r)} |")
    return out + [""]


def dataset_section() -> list[str]:
    counts = {r["variant"]: r["rows"] for r in read(PACKAGE / "data" / "final" / "Final_Paired_Email_Dataset_Summary.csv")}
    out = ["## Dataset", "",
           f"- Legitimate emails: {int(counts.get('legitimate', 0)):,}",
           f"- Phishing pairs: {int(counts.get('original_phishing', 0)):,} original emails and {int(counts.get('enhanced_phishing', 0)):,} LLM-enhanced versions",
           "- Data files compared with their frozen SHA-256 checksums:"]
    for rel, expected in DATA_SHA256.items():
        path = PACKAGE / rel
        status = "missing" if not path.exists() else ("OK" if hashlib.sha256(path.read_bytes()).hexdigest() == expected else "DIFFERENT")
        out.append(f"  - {status}: {rel}")
    return out + [""]


def report(src: Source, title: str) -> list[str]:
    lines = [f"# {title}", ""]
    lines += dataset_section()
    lines += rq12_section(src, "RQ1", "RQ1: Classical machine learning (primary analysis)")
    lines += rq12_section(src, "RQ2", "RQ2: Deep learning (primary analysis)")
    lines += rq3_section(src)
    lines += table1(src)
    lines += table2(src, "d_prime", "Table 2: Change in d′, primary and sensitivity analyses", 3)
    lines += table2(src, "roc_auc", "Change in AUC-ROC, primary and sensitivity analyses (complementary metric)", 5)
    lines += outlier_section(src)
    return lines


def same(a: dict | None, b: dict | None) -> str:
    if a is None or b is None:
        return "not run"
    keys = ("mean_difference", "ci95_low", "ci95_high", "p_value_two_sided")
    gap = max(abs(float(a[k]) - float(b[k])) for k in keys)
    return "match" if gap <= TOLERANCE else f"differs (up to {gap:.2g})"


def comparison(rerun: Source) -> list[str]:
    reported = Source()
    out = ["# Check: rerun compared with the reported results", "",
           "| Analysis | Metric | RQ1 | RQ2 | RQ3 |", "|---|---|---|---|---|"]
    for folder, label in ANALYSES:
        for metric, name in (("d_prime", "d′"), ("roc_auc", "AUC-ROC")):
            cells = [same(reported.ttest(rq, folder, metric), rerun.ttest(rq, folder, metric)) for rq in RQ_DIRS]
            out.append(f"| {label} | {name} | " + " | ".join(cells) + " |")
    rep = {(r["rq"], r["metric"], r["rule"]): r for r in read(reported.outlier_file())}
    new = {(r["rq"], r["metric"], r["rule"]): r for r in read(rerun.outlier_file())}
    out.append(f"| Outlier check | all rules | {' | '.join(', '.join(sorted({same(rep[k], new.get(k)) for k in rep if k[0] == rq})) for rq in RQ_DIRS)} |")
    out += ["", "RQ1 and RQ3 should match exactly. RQ2 runs on a GPU and can differ slightly on different hardware.", ""]
    return out


def blocks(lines: list[str]) -> list[tuple]:
    """Group report lines into headings, bullet lists, tables, and paragraphs."""
    out: list[tuple] = []
    for line in lines:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue  # Markdown separator row
            if out and out[-1][0] == "table":
                out[-1][1].append(cells)
            else:
                out.append(("table", [cells]))
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(("heading", level, line.lstrip("#").strip()))
        elif line.lstrip().startswith("- "):
            depth = (len(line) - len(line.lstrip())) // 2
            item = (depth, line.lstrip()[2:])
            if out and out[-1][0] == "list":
                out[-1][1].append(item)
            else:
                out.append(("list", [item]))
        elif line.strip():
            out.append(("text", line.lstrip("> ").replace("**", "")))
    return out


def as_text(lines: list[str]) -> str:
    """Plain text with aligned columns, for the terminal."""
    out: list[str] = []
    for block in blocks(lines):
        if block[0] == "heading":
            title = block[2]
            out += ["", title, ("=" if block[1] == 1 else "-") * len(title)]
        elif block[0] == "list":
            out += ["  " * depth + "- " + item for depth, item in block[1]]
        elif block[0] == "table":
            rows = block[1]
            widths = [max(len(r[i]) for r in rows if i < len(r)) for i in range(len(rows[0]))]
            fmt = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
            out += [fmt(rows[0]), "  ".join("-" * w for w in widths)] + [fmt(r) for r in rows[1:]]
        else:
            out += ["", block[1]]
    return "\n".join(out).strip() + "\n"


def as_html(lines: list[str], title: str) -> str:
    """A formatted page that opens in any web browser."""
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body: list[str] = []
    for block in blocks(lines):
        if block[0] == "heading":
            body.append(f"<h{block[1]}>{esc(block[2])}</h{block[1]}>")
        elif block[0] == "list":
            items = "".join(f"<li class='d{depth}'>{esc(item)}</li>" for depth, item in block[1])
            body.append(f"<ul>{items}</ul>")
        elif block[0] == "table":
            head, *rows = block[1]
            body.append("<table><thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in head) + "</tr></thead><tbody>"
                        + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>")
        else:
            body.append(f"<p>{esc(block[1])}</p>")
    style = ("body{font-family:Georgia,serif;max-width:1100px;margin:32px auto;padding:0 16px;line-height:1.5;color:#1d1d1d;background:#fff}"
             "h1{font-size:26px;border-bottom:2px solid #333;padding-bottom:6px}h2{font-size:20px;margin-top:32px}"
             "table{border-collapse:collapse;margin:8px 0;font-size:15px}th,td{padding:5px 12px;text-align:left;border-bottom:1px solid #ccc}"
             "thead th{border-top:2px solid #333;border-bottom:2px solid #333}tbody tr:last-child td{border-bottom:2px solid #333}"
             "li.d1{margin-left:24px;list-style:circle}p{color:#444}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{style}</style></head><body>{''.join(body)}</body></html>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", type=Path, default=None, metavar="WORKSPACE", help="rerun workspace created by run_all.py")
    parser.add_argument("--quick-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.compare is None:
        title = "Reported Results (Chapter 4)"
        lines = report(Source(), title)
        saved = PACKAGE / "RESULTS_SUMMARY.html"
    else:
        workspace = args.compare.resolve()
        if not (workspace / "rq_scripts").exists():
            raise SystemExit(f"No rerun outputs found in {workspace}")
        title = "Rerun Results"
        lines = report(Source(workspace), title)
        if args.quick_test:
            lines[1:1] = ["Quick test with 2 iterations per analysis. It shows that the code runs; the numbers are not expected to match Chapter 4.", ""]
        lines += comparison(Source(workspace))
        saved = workspace / "RERUN_RESULTS_SUMMARY.html"

    print(as_text(lines))
    saved.write_text(as_html(lines, title), encoding="utf-8")
    print(f"Saved (opens in any web browser): {saved}")


if __name__ == "__main__":
    main()
