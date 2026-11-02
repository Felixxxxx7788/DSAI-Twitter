"""
Investigate the 'Irrelevant' class (label 2).

The dataset documentation describes an ENTITY-LEVEL task ("Given a message and an
entity, judge the sentiment of the message about the entity") with 4 classes where
2 = Irrelevant. But the delivered CSV contains no entity column. If label 2 really
means "not about the entity", the label is unrecoverable from text alone - which
would explain why it scores worst. Test that hypothesis here.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
tr = pd.read_csv(ROOT / "data" / "processed" / "train_clean.csv")

print("=" * 96)
print("1. RAW FILE SCHEMA (does an entity column exist?)")
print("=" * 96)
raw = pd.read_csv(ROOT / "Data 5 - Twitter" / "Data 5 - Twitter" / "traindata.csv", encoding="utf-8-sig")
raw_te = pd.read_csv(ROOT / "Data 5 - Twitter" / "Data 5 - Twitter" / "testdata.csv", encoding="utf-8-sig")
print(f"  traindata.csv columns: {list(raw.columns)}")
print(f"  testdata.csv  columns: {list(raw_te.columns)}")
print("  -> No 'entity' / 'Topic' column in either file.")

print("\n" + "=" * 96)
print("2. IS LABEL 2 SEPARABLE BY TOPIC? Compare brand-mention rate per class")
print("=" * 96)
BRANDS = ["PUBG", "Amazon", "Verizon", "CallofDuty", "Call of Duty", "Xbox", "PlayStation",
          "Microsoft", "Fortnite", "Overwatch", "NBA2K", "FIFA", "Madden", "Google",
          "Apple", "Facebook", "Twitter", "Rockstar", "GTA", "Blizzard", "Nintendo"]
brand_re = "|".join(re.escape(b) for b in BRANDS)
rows = []
for lab, name in [(0, "Negative"), (1, "Positive"), (2, "Irrelevant"), (3, "Neutral")]:
    sub = tr[tr["sentiment"] == lab]["phrase"].astype(str)
    rows.append({
        "class": name, "n": len(sub),
        "% mention a brand": round(sub.str.contains(brand_re, case=False, regex=True).mean() * 100, 1),
        "% first-person (i/my/me)": round(sub.str.contains(r"\b(?:i|my|me|im)\b", case=False, regex=True).mean() * 100, 1),
        "% has profanity": round(sub.str.contains(r"\b(?:fuck|shit|damn|hell|ass)\w*", case=False, regex=True).mean() * 100, 1),
        "% has news words": round(sub.str.contains(r"\b(?:news|report|announce|launch|update|says)\w*", case=False, regex=True).mean() * 100, 1),
    })
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 96)
print("3. WHAT DOES LABEL 2 LOOK LIKE? (10 random samples)")
print("=" * 96)
for _, r in tr[tr["sentiment"] == 2].sample(10, random_state=7).iterrows():
    print(f"  - {str(r['phrase'])[:130]}")

print("\n" + "=" * 96)
print("4. LABEL 2 vs OTHERS: token overlap (Jaccard of top-50 vocab)")
print("=" * 96)


def top_vocab(lab: int, k: int = 50) -> set[str]:
    c = Counter(w for s in tr.loc[tr["sentiment"] == lab, "text_lemma"] for w in str(s).split())
    return {w for w, _ in c.most_common(k)}


v = {lab: top_vocab(lab) for lab in range(4)}
for a in range(4):
    for b in range(a + 1, 4):
        j = len(v[a] & v[b]) / len(v[a] | v[b])
        print(f"  {a} vs {b}: Jaccard={j:.3f}")

print("\n" + "=" * 96)
print("5. HOW OFTEN IS 'Irrelevant' CONFUSED WITH A SENTIMENT CLASS? (baseline preds)")
print("=" * 96)
preds = pd.read_csv(ROOT / "reports" / "tables" / "test_predictions.csv")
print(preds[preds["actual"] == "Irrelevant"]["predicted"].value_counts().to_string())
print(f"\n  Only {int(((preds['actual']=='Irrelevant') & (preds['predicted']=='Irrelevant')).sum())} "
      f"of {int((preds['actual']=='Irrelevant').sum())} 'Irrelevant' rows were correctly identified.")
