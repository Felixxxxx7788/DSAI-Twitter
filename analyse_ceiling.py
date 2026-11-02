"""
Ablation: what is the empirically achievable ceiling, and how much of it is label noise?

Runs on TRAIN with 5-fold CV only (no test peeking) and answers three questions:

1. `oracle-human` ceiling : fit on most of a fold, evaluate on the fold. This is the
   normal CV number (~0.49). It is NOT a ceiling, it is a sample-noise estimate.
2. `self-training` recovery: take the model's own confident predictions on train,
   retrain on them and re-measure. If accuracy against the *original* labels barely
   moves while agreement with the model's own labels rockets, the labels - not the
   features - are the binding constraint.
3. `leave-one-out neighbour consistency`: for each train row, how often do its 10
   nearest TF-IDF neighbours share its label? A high value means the label is
   recoverable from the text; a low value means the same text is labelled
   inconsistently (irreducible noise).

Run: python src/analyse_ceiling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "reports" / "tables"

LABELS = ["Negative", "Positive", "Irrelevant", "Neutral"]
COLS = ["phrase", "text", "text_lemma"]


def make_pipe(col: str) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95,
                                  sublinear_tf=True, strip_accents="unicode",
                                  token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")),
        ("clf", LogisticRegression(C=1.0, max_iter=3000)),
    ])


def oof_proba(pipe, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    proba = np.zeros((len(y), 4))
    pred = np.zeros(len(y), dtype=int)
    for tr, va in skf.split(X, y):
        p = pipe.__class__(pipe.steps)
        import copy
        p = copy.deepcopy(pipe)
        p.fit(X.iloc[tr], y[tr])
        proba[va] = p.predict_proba(X.iloc[va])
        pred[va] = proba[va].argmax(1)
    return proba, pred


def main():
    train = pd.read_csv(PROC / "train_clean.csv")
    y = train["sentiment"].astype(int).values
    out = {}

    print("=" * 90)
    print("1. Neighbour-consistency: can the label be recovered from the text at all?")
    print("=" * 90)
    neigh_rows = []
    for col in COLS:
        X = train[col].fillna("").astype(str)
        Xv = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                             strip_accents="unicode").fit_transform(X)
        nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(Xv)
        _, idx = nn.kneighbors(Xv)
        same = np.array([np.mean(y[idx[i, 1:]] == y[i]) for i in range(len(y))])
        any_same = np.array([np.any(y[idx[i, 1:]] == y[i]) for i in range(len(y))])
        out[f"nn10_label_agreement_{col}"] = round(float(same.mean()), 3)
        out[f"nn10_has_any_agreeing_neighbour_{col}"] = round(float(any_same.mean()), 3)
        neigh_rows.append({"column": col, "nn10_same_label_fraction": same.mean(),
                           "nn10_any_same_label": any_same.mean(),
                           "cosine_sim_median": float(
                               np.median(nn.kneighbors(Xv, n_neighbors=2)[0][:, 1]))})
        print(f"  {col:<12s} 10-NN same-label fraction = {same.mean():.3f}   "
              f"any agreeing neighbour = {any_same.mean()*100:.1f}%")
    pd.DataFrame(neigh_rows).to_csv(TAB / "ceiling_neighbour_consistency.csv", index=False)

    print("\n" + "=" * 90)
    print("2. Self-training recovery test (does the model agree with itself?)")
    print("=" * 90)
    col = "phrase"
    X = train[col].fillna("").astype(str)
    proba, pred = oof_proba(make_pipe(col), X, y)
    out["oof_accuracy_vs_human_labels"] = round(float(accuracy_score(y, pred)), 3)

    # retrain on the model's own OOF predictions, then re-measure against the
    # original labels and against the pseudo-labels
    pipe = make_pipe(col)
    pipe.fit(X, pred)
    pred2 = pipe.predict(X)
    out["self_trained_accuracy_vs_human_labels"] = round(float(accuracy_score(y, pred2)), 3)
    out["self_trained_accuracy_vs_pseudolabels"] = round(float(accuracy_score(pred, pred2)), 3)
    print(f"  OOF accuracy vs human labels            : {out['oof_accuracy_vs_human_labels']:.3f}")
    print(f"  self-trained accuracy vs human labels   : {out['self_trained_accuracy_vs_human_labels']:.3f}")
    print(f"  self-trained accuracy vs own pseudo-labels: {out['self_trained_accuracy_vs_pseudolabels']:.3f}")

    print("\n" + "=" * 90)
    print("3. Agreement statistics: how noisy is the label, in kappa terms?")
    print("=" * 90)
    kappa = cohen_kappa_score(y, pred, weights=None)
    # ordinal only makes sense on a Neg-Pos scale, do it on classes 0,1,3
    ord_mask = np.isin(y, [0, 1, 3]) & np.isin(pred, [0, 1, 3])
    out["cohen_kappa_4class"] = round(float(kappa), 3)
    out["within1_accuracy"] = round(float((np.abs(y - pred) <= 1).mean()), 3)
    print(f"  Cohen kappa (4 classes)  : {out['cohen_kappa_4class']:.3f}")
    print(f"  within-1 accuracy        : {out['within1_accuracy']*100:.1f}%")

    print("\n" + "=" * 90)
    print("4. Negative vs Positive only (the pure polarity signal)")
    print("=" * 90)
    m = np.isin(y, [0, 1])
    Xp = X[m].reset_index(drop=True)
    yp = (y[m] == 1).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, va in skf.split(Xp, yp):
        p = make_pipe(col)
        p.fit(Xp.iloc[tr], yp[tr])
        accs.append(accuracy_score(yp[va], p.predict(Xp.iloc[va])))
    out["binary_pos_neg_cv_accuracy"] = round(float(np.mean(accs)), 3)
    out["binary_pos_neg_majority"] = round(float(max(yp.mean(), 1 - yp.mean())), 3)
    print(f"  Pos-vs-Neg 5-fold CV accuracy = {np.mean(accs):.3f} "
          f"(majority = {max(yp.mean(), 1-yp.mean()):.3f})")

    print("\n" + "=" * 90)
    print("5. Without the unlearnable Irrelevant class")
    print("=" * 90)
    m3 = y != 2
    X3 = X[m3].reset_index(drop=True)
    y3 = y[m3]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs3 = []
    for tr, va in skf.split(X3, y3):
        p = make_pipe(col)
        p.fit(X3.iloc[tr], y3[tr])
        accs3.append(accuracy_score(y3[va], p.predict(X3.iloc[va])))
    out["three_class_cv_accuracy"] = round(float(np.mean(accs3)), 3)
    print(f"  3-class (no Irrelevant) 5-fold CV accuracy = {np.mean(accs3):.3f}")

    json.dump(out, open(PROC / "ceiling_analysis.json", "w"), indent=2)
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
