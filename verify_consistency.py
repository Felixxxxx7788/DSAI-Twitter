"""Final consistency check across the three deliverables (report, layman doc, notebook)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import docx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
met = json.loads((ROOT / "models" / "baseline_metrics.json").read_text(encoding="utf-8"))
eda = json.loads((ROOT / "data" / "processed" / "eda_summary.json").read_text(encoding="utf-8"))

# canonical values
canon = {
    "test_accuracy": round(met["test_accuracy"] * 100, 1),
    "test_macro_f1": round(met["test_macro_f1"] * 100, 1),
    "cv_macro_f1": round(met["cv"][3]["macro_f1_mean"] * 100, 1),
    "balanced_cv_f1": round(met["cv"][4]["macro_f1_mean"] * 100, 1),
    "majority": round(met["majority_baseline_accuracy"] * 100, 1),
    "train_rows": eda["train_rows_out"],
    "test_rows": eda["test_rows_out"],
}
print("CANONICAL VALUES (from JSON artefacts)")
for k, v in canon.items():
    print(f"  {k:<18} {v}")


def docx_text(p: Path) -> str:
    d = docx.Document(p)
    parts = [pp.text for pp in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                parts.append(c.text)
    return "\n".join(parts)


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


print("\n" + "=" * 74)
print("REPORT")
print("=" * 74)
rt = docx_text(ROOT / "reports" / "Interim_Report_Group8_Twitter_Sentiment.docx")
rn = numbers_in(rt)
checks = {
    "49.6 (test accuracy)": "49.6" in rn,
    "44.0 (test macro F1)": "44.0" in rn,
    "39.4 (CV macro F1)": "39.4" in rn,
    "46.0 (balanced CV F1)": "46.0" in rn,
    "30.2 (majority)": "30.2" in rn,
    "592 (train rows)": "592" in rn,
}
for label, ok in checks.items():
    print(f"  {'OK ' if ok else 'MISS'} {label}")

print("\n" + "=" * 74)
print("LAYMAN DOC (Cantonese)")
print("=" * 74)
lt = docx_text(ROOT / "reports" / "白話解釋手冊_Group8_Twitter情緒分析.docx")
ln = numbers_in(lt)
checks_l = {
    "49.6 (test accuracy)": "49.6" in ln,
    "44.0 (test macro F1)": "44.0" in ln,
    "39.4 (CV macro F1)": "39.4" in ln,
    "30.2 (majority)": "30.2" in ln,
    "596 (raw train)": "596" in ln,
    "592 (clean train)": "592" in ln,
}
for label, ok in checks_l.items():
    print(f"  {'OK ' if ok else 'MISS'} {label}")

print("\n" + "=" * 74)
print("NOTEBOOK (executed outputs)")
print("=" * 74)
nb = json.loads((ROOT / "DSAI4205_Interim_Group8_Twitter_Sentiment.ipynb").read_text(encoding="utf-8"))
nb_out = []
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    for o in c.get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain")
        if t:
            nb_out.append("".join(t))
nb_text = "\n".join(nb_out)
checks_n = {
    "49.62 (test accuracy)": "49.62" in nb_text,
    "44.03 (test macro F1)": "44.03" in nb_text,
    "30.23 (majority)": "30.23" in nb_text,
    "200 misclassified": "200 of 397" in nb_text,
    "592 / 397 rows": "592" in nb_text and "397" in nb_text,
}
for label, ok in checks_n.items():
    print(f"  {'OK ' if ok else 'MISS'} {label}")

print("\n" + "=" * 74)
print("ROW COUNTS")
print("=" * 74)
print(f"  eda_summary.json  : train {eda['train_rows_in']} -> {eda['train_rows_out']}, "
      f"test {eda['test_rows_in']} -> {eda['test_rows_out']}")
print(f"  baseline_metrics  : train {met['train_rows']}, test {met['test_rows']}")
print(f"  consistent        : {met['train_rows'] == eda['train_rows_out'] and met['test_rows'] == eda['test_rows_out']}")
