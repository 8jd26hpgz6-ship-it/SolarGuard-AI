"""Summarise outputs/ produced by solarguard_train.py into paper-ready tables.

Usage:  python summarize_results.py --output-dir outputs

Reads all_results.csv and each run's classification_report.json; writes
summary_tables.md (and prints it). Mean +/- sample SD is over the seeds that
were actually run; nothing is estimated or filled in.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

CONFIGS = ["CNN_CE", "CNN_CBAM_CE", "CNN_Focal", "CNN_CBAM_Focal"]
MINORITY = ["Soiling", "Diode-Multi", "Hot-Spot-Multi"]


def ms(values):
    v = np.asarray(values, dtype=float)
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    return f"{v.mean():.4f} ± {sd:.4f}" if len(v) > 1 else f"{v.mean():.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs")
    out = Path(ap.parse_args().output_dir)
    with (out / "all_results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = defaultdict(list)
    for r in rows:
        by[(r["configuration"], r["condition"])].append(r)
    configs = [c for c in CONFIGS if any(k[0] == c for k in by)]
    conditions = sorted({k[1] for k in by}, key=lambda c: (c != "clean_0", c))
    lines = []

    lines += ["## Table A. Clean test set (mean ± SD over seeds)", "",
              "| Configuration | Seeds | Accuracy | Macro-F1 | Macro-recall | Best val Macro-F1 |",
              "|---|---|---|---|---|---|"]
    for c in configs:
        g = by[(c, "clean_0")]
        lines.append(f"| {c} | {len(g)} | {ms([float(r['accuracy']) for r in g])} | "
                     f"{ms([float(r['macro_f1']) for r in g])} | {ms([float(r['macro_recall']) for r in g])} | "
                     f"{ms([float(r['best_validation_macro_f1']) for r in g])} |")

    lines += ["", "## Table B. Minority-class recall, clean test set (mean ± SD)", "",
              "| Configuration | " + " | ".join(MINORITY) + " |", "|---|" + "---|" * len(MINORITY)]
    for c in configs:
        cells = []
        for cls in MINORITY:
            vals = []
            for r in by[(c, "clean_0")]:
                rep = json.loads((out / c / f"seed_{r['seed']}" / "clean_0" / "classification_report.json").read_text())
                vals.append(rep[cls]["recall"])
            cells.append(ms(vals))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    lines += ["", "## Table C. Macro-F1 under perturbation (mean over seeds)", "",
              "| Condition | " + " | ".join(configs) + " |", "|---|" + "---|" * len(configs)]
    for cond in conditions:
        lines.append(f"| {cond} | " + " | ".join(
            f"{np.mean([float(r['macro_f1']) for r in by[(c, cond)]]):.4f}" if by[(c, cond)] else "-"
            for c in configs) + " |")

    text = "\n".join(lines) + "\n"
    (out / "summary_tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
