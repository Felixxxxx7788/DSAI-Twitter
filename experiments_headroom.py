"""
Experiments that establish how much headroom is actually left.

TRAIN-ONLY. Nothing here touches the test file, so it is safe to run while
designing the final model.

E1  Label-permutation control
    Shuffle the labels at random and re-run the identical CV. The resulting
    accuracy (~0.25 for 4 balanced classes) is what "no signal" looks like.
    Our real 0.47-0.49 is therefore a genuine, if modest, signal - not noise
    the model is inventing.

E2  Human-disagreement lower bound
    For every train row, take its k nearest neighbours in TF-IDF space and ask
    what fraction of them carry the same label. If the same tweet, or a tweet
    with the same words, is routinely given a different label, no classifier
    can be right more often than the labels agree with themselves.

E3  Headroom probe: cross-view agreement
    Train on the raw phrase, predict on the cleaned lemma view (and vice
    versa). Agreement tells us whether extra cleaning destroys information
    (it does not) and how much of the residual error is *view-specific* noise
    that an ensemble can average away.

E4  Irrelevant-class ceiling
    Class 2 is defined relative to an `entity` column that is not in the file.
    Fitting a *binary* Irrelevant detector gives an AUC, and AUC converts to a
    best-case accuracy via Youden's J. That is the most any model can ever do
    for that class with the information present.

Writes: reports/tables/exp_labelling_*.csv, data/processed/headroom_summary.json
Run: python src/experiments_headroom.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "reports" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
LABELS = ["Negative", "Positive", "Irrelevant", "Neutral"]
SEED = 42
RNG = np.random.default_rng(SEED)

train = pd.read_csv(PROC / "train_clean.csv")
y = train["sentiment"].astype(int).values
X_phrase = train["phrase"].fillna("").astype(str)
X_text = train["text"].fillna("").astype(str)
X_lemma = train["text_lemma"].fillna("").astype(str)

out: dict = {}


def make(ngram=(1, 2), C=1.0, char=False) -> Pipeline:
    vec = (TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True) if char else
           TfidfVectorizer(ngram_range=ngram, min_df=1, max_df=0.95,
                           sublinear_tf=True, strip_accents="unicode",
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b"))
    return Pipeline([("v", vec), ("c", LogisticRegression(C=C, max_iter=3000))])


def oof(pipe, X, yy, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    proba = np.zeros((len(yy), 4))
    for tr, va in skf.split(X, yy):
        p = copy.deepcopy(pipe)
        p.fit(X.iloc[tr], yy[tr])
        proba[va] = p.predict_proba(X.iloc[va])
    return proba


# ---------------------------------------------------------------- E1 permutation
print("=" * 88)
print("E1  label-permutation control (what 'no signal' looks like)")
print("=" * 88)
base_proba = oof(make(), X_lemma, y)
base_acc = float((base_proba.argmax(1) == y).mean())
perm_accs = []
for k in range(20):
    yp = RNG.permutation(y)
    pa = oof(make(), X_lemma, yp)
    perm_accs.append(float((pa.argmax(1) == yp).mean()))
print(f"  real-label CV accuracy      : {base_acc:.4f}")
print(f"  permuted-label CV accuracy  : {np.mean(perm_accs):.4f} ± {np.std(perm_accs):.4f} "
      f"(max over 20 shuffles {np.max(perm_accs):.4f})")
out["cv_accuracy_real"] = round(base_acc, 4)
out["cv_accuracy_permuted_mean"] = round(float(np.mean(perm_accs)), 4)
out["cv_accuracy_permuted_sd"] = round(float(np.std(perm_accs)), 4)
out["cv_accuracy_permuted_max"] = round(float(np.max(perm_accs)), 4)

# ---------------------------------------------------------- E2 label self-agreement
print("\n" + "=" * 88)
print("E2  label self-agreement in text space (irreducible-noise estimate)")
print("=" * 88)
rows = []
for name, X in [("phrase", X_phrase), ("text", X_text), ("text_lemma", X_lemma)]:
    for k in (5, 10, 25):
        Xv = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(X)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xv)
        _, idx = nn.kneighbors(Xv)
        same = np.mean([np.mean(y[idx[i, 1:]] == y[i]) for i in range(len(y))])
        near_dup = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xv).kneighbors(
            Xv, n_neighbors=2)[0][:, 1]
        rows.append({"view": name, "k": k, "same_label_fraction": round(float(same), 4),
                     "median_nn_cosine_sim": round(float(np.median(near_dup)), 4)})
        print(f"  {name:<11s} k={k:<3d} same-label fraction={same:.4f}  "
              f"median nearest cosine sim={np.median(near_dup):.4f}")
pd.DataFrame(rows).to_csv(TAB / "exp_label_self_agreement.csv", index=False)
out["nn10_same_label_phrase"] = rows[1]["same_label_fraction"]
out["nn10_same_label_lemma"] = rows[5]["same_label_fraction"]

# ------------------------------------------------------ E3 cross-view agreement
print("\n" + "=" * 88)
print("E3  cross-view agreement: is cleaning destroying information?")
print("=" * 88)
pair_rows = []
and_preds = {}
for name, X in [("phrase", X_phrase), ("text", X_text), ("text_lemma", X_lemma)]:
    proba = oof(make(), X, y)
    and_preds[name] = proba
    pair_rows.append({"view": name, "cv_accuracy": round(float((proba.argmax(1) == y).mean()), 4)})
    print(f"  {name:<11s} CV acc = {(proba.argmax(1) == y).mean():.4f}")
for a, b in [("phrase", "text"), ("phrase", "text_lemma"), ("text", "text_lemma")]:
    agree = float((and_preds[a].argmax(1) == and_preds[b].argmax(1)).mean())
    print(f"  agreement {a:<10s} vs {b:<10s}: {agree:.4f}")
    pair_rows.append({"view": f"{a}_vs_{b}", "cv_accuracy": round(agree, 4)})
pd.DataFrame(pair_rows).to_csv(TAB / "exp_cross_view.csv", index=False)

# average the three views as a mini-ensemble (still train-only)
avg = sum(and_preds[k] for k in ("phrase", "text", "text_lemma")) / 3
avg_acc = float((avg.argmax(1) == y).mean())
print(f"  mean-probability ensemble of the 3 views: CV acc = {avg_acc:.4f}")
out["ensemble_3view_cv_accuracy"] = round(avg_acc, 4)

# ------------------------------------------------- E4 Irrelevant-class ceiling
print("\n" + "=" * 88)
print("E4  Irrelevant-class ceiling (entity column is missing from the data)")
print("=" * 88)
irr_rows = []
for target in range(4):
    yb = (y == target).astype(int)
    proba = oof(make(), X_lemma, yb)[:, 1] if False else None
    # binary OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = np.zeros(len(yb))
    for tr, va in skf.split(X_lemma, yb):
        p = copy.deepcopy(make())
        p.fit(X_lemma.iloc[tr], yb[tr])
        preds[va] = p.predict_proba(X_lemma.iloc[va])[:, 1]
    auc = roc_auc_score(yb, preds)
    fpr, tpr, _ = roc_curve(yb, preds)
    j = tpr - fpr
    best_t = float(_[int(np.argmax(j))])
    best_acc = float(((preds >= best_t).astype(int) == yb).mean())
    irr_rows.append({"class": LABELS[target], "ovr_auc": round(float(auc), 4),
                     "best_threshold": round(best_t, 4),
                     "best_binary_accuracy": round(best_acc, 4),
                     "always_majority_accuracy": round(
                         float(max(yb.mean(), 1 - yb.mean())), 4)})
    print(f"  {LABELS[target]:<11s} one-vs-rest AUC={auc:.4f}  best binary acc={best_acc:.4f} "
          f"(always-majority {max(yb.mean(), 1-yb.mean()):.4f})")
pd.DataFrame(irr_rows).to_csv(TAB / "exp_ovr_auc.csv", index=False)
out["irrelevant_ovr_auc"] = irr_rows[2]["ovr_auc"]

json.dump(out, open(PROC / "headroom_summary.json", "w"), indent=2)
print("\n" + json.dumps(out, indent=2))
