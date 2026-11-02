"""
Label-noise forensics for the Twitter sentiment dataset.

Quantifies, with reproducible numbers, the three label pathologies identified in
the interim diagnosis:

1. Lexicon contradiction  - tweets whose polarity words point one way but whose
   label points another (TextBlob-free: uses a small built-in polar lexicon plus
   VADER if available).
2. Same-text / same-document disagreement - identical or near-identical text
   carrying different labels.
3. Irrelevant unlearnability - one-vs-rest separability of class 2 from the rest.

Writes reports/tables/label_noise_audit.csv and label_noise_summary.json.

Run: python src/audit_labels.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "reports" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

LABELS = ["Negative", "Positive", "Irrelevant", "Neutral"]

POS_WORDS = set("""amazing awesome best better beautiful brilliant cool delighted enjoy enjoyed
excellent excited exciting fantastic fine fun glad good great happy helpful hope impressive
incredible interesting like liked lovely love loved nice perfect pleased recommend satisfied
super sweet thanks thank thankful terrific thanks wonderful wow win winner yay yes excellent
favorite fave <3 :) :d :p lol haha happy amazing""".split())
NEG_WORDS = set("""angry annoyed annoying awful bad broken bullshit can't cant crap crash crashed
crap disgusting disappointed disappointment dislike dumb fail failed failure fake
frustrated frustrating hate hated horrible hurt issue lame mad mess miserable need
never nobody not poorest problem rude sad scared shit shitty sorry stupid sucks suck sucky
terrible tired trash ugly unable unhappy useless waste worst worthless wrong wtf :( >:(""".split())

TAG_RE = re.compile(r"@\s?\w+|http\S+|#\w+")


def vader_scores(texts):
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return np.array([sia.polarity_scores(t)["compound"] for t in texts])
    except Exception:
        return None


def lexicon_count(text):
    toks = re.findall(r"[a-z']+", str(text).lower())
    pos = sum(t in POS_WORDS for t in toks)
    neg = sum(t in NEG_WORDS for t in toks)
    return pos, neg


def main():
    train = pd.read_csv(PROC / "train_clean.csv")
    test = pd.read_csv(PROC / "test_clean.csv")
    train["label"] = train["sentiment"].map(dict(enumerate(LABELS)))
    test["label"] = test["sentiment"].map(dict(enumerate(LABELS)))
    both = pd.concat([train.assign(split="train"), test.assign(split="test")],
                     ignore_index=True)

    summary: dict = {"n_train": len(train), "n_test": len(test)}

    # ---------------------------------------------------------------- 1. lexicon
    v = vader_scores(both["phrase"].astype(str).tolist())
    both["vader"] = v if v is not None else np.nan
    pos_neg = both["phrase"].astype(str).map(lexicon_count)
    both["n_pos"], both["n_neg"] = [p for p, _ in pos_neg], [n for _, n in pos_neg]
    both["lexi"] = both["n_pos"] - both["n_neg"]

    def band(x):
        if x >= 2:
            return "clearly positive"
        if x <= -2:
            return "clearly negative"
        return "mixed / neutral"

    both["lexi_band"] = both["lexi"].map(band)
    summary["pol_lexicon_rows"] = int((both["lexi_band"] != "mixed / neutral").sum())
    agree = pd.crosstab(both["lexi_band"], both["label"])
    agree.to_csv(TAB / "label_noise_lexicon_band.csv")
    print("\nLexicon band x label (all rows)")
    print(agree.to_string())

    pos_rows = both[both["lexi_band"] == "clearly positive"]
    neg_rows = both[both["lexi_band"] == "clearly negative"]
    summary["clearly_positive_rows"] = int(len(pos_rows))
    summary["clearly_positive_labelled_positive"] = int((pos_rows["label"] == "Positive").sum())
    summary["clearly_positive_pct_positive"] = round(
        (pos_rows["label"] == "Positive").mean() * 100, 1) if len(pos_rows) else None
    summary["clearly_negative_rows"] = int(len(neg_rows))
    summary["clearly_negative_labelled_negative"] = int((neg_rows["label"] == "Negative").sum())
    summary["clearly_negative_pct_negative"] = round(
        (neg_rows["label"] == "Negative").mean() * 100, 1) if len(neg_rows) else None
    print(f"\nClearly-positive tweets: {summary['clearly_positive_rows']}, "
          f"labelled Positive: {summary['clearly_positive_pct_positive']}%")
    print(f"Clearly-negative tweets: {summary['clearly_negative_rows']}, "
          f"labelled Negative: {summary['clearly_negative_pct_negative']}%")

    pos_vs_neg = both[both["label"].isin(["Positive", "Negative"])].copy()
    if v is not None and pos_vs_neg["vader"].notna().any():
        pv = pos_vs_neg["vader"]
        pv = pv.fillna(0)
        sign_agree = ((pv > 0) & (pos_vs_neg["label"] == "Positive")) | \
                     ((pv < 0) & (pos_vs_neg["label"] == "Negative"))
        summary["vader_agreement_pos_neg"] = round(float(sign_agree.mean()) * 100, 1)
        print(f"VADER sign agrees with Positive/Negative label: "
              f"{summary['vader_agreement_pos_neg']}%")

    # ------------------------------------------------- 2. duplicate-text conflicts
    norm = both["phrase"].astype(str).str.lower().map(lambda s: TAG_RE.sub(" ", s))
    norm = norm.map(lambda s: re.sub(r"[^a-z0-9 ]+", " ", s))
    norm = norm.map(lambda s: re.sub(r"\s+", " ", s).strip())
    both["norm"] = norm
    grp = both.groupby("norm")["label"].agg(["nunique", "count", "first"])
    conflicts = grp[grp["nunique"] > 1]
    summary["exact_text_conflicts"] = int(len(conflicts))
    summary["rows_in_conflicts"] = int(conflicts["count"].sum())
    conflicts.sort_values("count", ascending=False).head(30).to_csv(
        TAB / "label_noise_conflicts.csv")
    print(f"\nExact-text label conflicts: {len(conflicts)} unique texts "
          f"({int(conflicts['count'].sum())} rows)")

    # ------------------------------------------------ 3. Irrelevant separability
    txt = both["text_lemma"].fillna("")
    y = both["sentiment"].astype(int).values
    for target in range(4):
        yb = (y == target).astype(int)
        pipe_y = yb
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        Xv = vec.fit_transform(txt)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pp = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), Xv, pipe_y,
                               cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        auc = roc_auc_score(pipe_y, pp)
        summary[f"ovr_auc_{LABELS[target].lower()}"] = round(float(auc), 3)
    print("\nOne-vs-rest AUC (0.5 = no learnable signal):")
    for k in LABELS:
        print(f"  {k:<11s} {summary['ovr_auc_' + k.lower()]}")

    # ---------------------------------------------------------- 4. example list
    examples = []
    for lbl in ["Positive", "Negative", "Neutral", "Irrelevant"]:
        sub = both[(both["label"] == lbl) & (both["lexi_band"] != "mixed / neutral")]
        sub = sub[((sub["lexi_band"] == "clearly positive") & (lbl != "Positive")) |
                  ((sub["lexi_band"] == "clearly negative") & (lbl != "Negative"))]
        for _, r in sub.head(12).iterrows():
            examples.append({"phrase": r["phrase"], "label": lbl,
                             "lexicon_band": r["lexi_band"], "vader": r.get("vader")})
    pd.DataFrame(examples).to_csv(TAB / "label_noise_examples.csv", index=False)

    json.dump(summary, open(PROC / "label_noise_summary.json", "w"), indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
