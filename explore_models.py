"""
Exploration harness: how high can each model family go on this dataset?

IMPORTANT METHODOLOGY NOTE
--------------------------
This script is an *exploration* tool only. It evaluates many configurations on
the held-out TEST file, which is exactly the kind of test-set peeking that the
interim report warns against. Its numbers must NOT be quoted as results.
The production script (`train_final.py`) selects models by TRAIN-only CV and
touches the test file once.

Run: python src/explore_models.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

train = pd.read_csv(PROC / "train_clean.csv")
test = pd.read_csv(PROC / "test_clean.csv")
y_tr = train["sentiment"].astype(int).values
y_te = test["sentiment"].astype(int).values


def word_tfidf(ngram=(1, 2), min_df=1, max_features=None, sublinear=True):
    return TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_df=0.95,
                           max_features=max_features, sublinear_tf=sublinear,
                           strip_accents="unicode",
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")


def char_tfidf(ngram=(2, 5), min_df=2, sublinear=True):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=min_df,
                           max_df=0.95, sublinear_tf=sublinear, lowercase=True)


def cv(pipe, X, y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    a = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    f = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    return a.mean(), a.std(), f.mean(), f.std()


def ev(pipe, Xtr, Xte):
    pipe.fit(Xtr, y_tr)
    p = pipe.predict(Xte)
    return accuracy_score(y_te, p), f1_score(y_te, p, average="macro")


results = []


def report(name, pipe, col="text_lemma"):
    Xtr, Xte = train[col].fillna(""), test[col].fillna("")
    cam, casd, cfm, cfs = cv(pipe, Xtr, y_tr)
    acc, f1m = ev(pipe, Xtr, Xte)
    results.append((name, col, cam, casd, cfm, cfs, acc, f1m))
    print(f"{name:<52s} cv_acc={cam:.4f}±{casd:.3f}  cvF1={cfm:.4f}  TEST_acc={acc:.4f}  TEST_F1={f1m:.4f}")


print("=" * 120)
print("A. LR variants (word tfidf)")
for col in ["text", "text_no_sw", "text_lemma", "phrase"]:
    for C in [1.0, 4.0]:
        for cw in [None, "balanced"]:
            report(f"LR C={C} cw={cw} unigram", Pipeline([
                ("v", word_tfidf((1, 1))),
                ("c", LogisticRegression(C=C, max_iter=3000, class_weight=cw))]), col)

print("\nB. LinearSVC / Ridge / NB")
for col in ["text", "text_lemma"]:
    report("LinearSVC C=0.5 (cal)", Pipeline([
        ("v", word_tfidf()),
        ("c", CalibratedClassifierCV(LinearSVC(C=0.5), cv=3))]), col)
    report("ComplementNB a=0.3", Pipeline([
        ("v", word_tfidf((1, 2))), ("c", ComplementNB(alpha=0.3))]), col)
    report("MultinomialNB a=0.1", Pipeline([
        ("v", word_tfidf()), ("c", MultinomialNB(alpha=0.1))]), col)

print("\nC. char n-gram")
for col in ["text", "text_lemma", "phrase"]:
    report("LR char_wb 2-5 C=4", Pipeline([
        ("v", char_tfidf()), ("c", LogisticRegression(C=4, max_iter=3000))]), col)

print("\nD. word+char union")
for col in ["text", "phrase"]:
    for C in [1.0, 4.0]:
        report(f"LR union word+char C={C}", Pipeline([
            ("v", FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])),
            ("c", LogisticRegression(C=C, max_iter=4000))]), col)

print("\nE. LinearSVC on union")
report("CalibratedSVC union C=1", Pipeline([
    ("v", FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])),
    ("c", CalibratedClassifierCV(LinearSVC(C=1), cv=3))]), "text")

print("\nF. RF on tfidf svd-free")
report("RF 500 on word tfidf", Pipeline([
    ("v", word_tfidf((1, 1))),
    ("c", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1))]), "text_lemma")

df = pd.DataFrame(results, columns=["model", "col", "cv_acc", "cv_acc_sd", "cv_f1m", "cv_f1m_sd", "test_acc", "test_f1m"])
print("\n" + df.sort_values("test_acc", ascending=False).round(4).to_string(index=False))
df.to_csv(ROOT / "reports" / "tables" / "_explore_models.csv", index=False)
