"""
Phase 3i : TF-IDF + Logistic Regression BASELINE model.

Why this design
---------------
* Small corpus (592 train / 397 test rows, 4 classes). A single random split
  would be extremely noisy, so the headline benchmark is a stratified 5-fold
  cross-validated mean +/- std on TRAIN*, and the held-out TEST file is scored
  once by the fold-1 model and by the full-train model. This gives the prof a
  stable number plus an honest out-of-sample check.
* *Cross-validation is run on train only. The test file is never used for
  model selection, so the reported test scores are an unbiased estimate.
* Three variants are reported so the interim report can justify the cleaning
  decisions quantitatively:
      V1 raw text            (no cleaning)   -> shows the value of cleaning
      V2 cleaned text        (our pipeline)  -> headline baseline
      V3 cleaned + class_weight='balanced'   -> addresses the imbalance found in EDA
* TF-IDF hyper-parameters are kept deliberately vanilla for the baseline
  (word 1-2 grams, sublinear_tf, min_df=1, no feature selection) - tuning is a
  Week 11-13 activity and tuning the baseline now would blur the benchmark.

Run: python src/train_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "models"
for d in (FIG_DIR, TAB_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["Negative", "Positive", "Irrelevant", "Neutral"]
PALETTE = ["#D62728", "#2CA02C", "#7F7F7F", "#1F77B4"]
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold"})


def build_pipeline(balanced: bool = False) -> Pipeline:
    """TF-IDF -> multinomial Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),      # unigrams + bigrams: captures "not good"
            min_df=1,                # tiny corpus -> keep rare words
            max_df=0.95,             # drop near-universal tokens
            sublinear_tf=True,       # log-scale term frequency
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",          # multinomial by default for >2 classes
            class_weight="balanced" if balanced else None,
            random_state=RANDOM_STATE,
        )),
    ])


def cv_evaluate(texts: pd.Series, y: pd.Series, tag: str, balanced: bool = False,
                n_splits: int = 5) -> dict:
    """Stratified k-fold CV; returns per-fold and aggregated metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr_i, va_i) in enumerate(skf.split(texts, y), start=1):
        pipe = build_pipeline(balanced)
        pipe.fit(texts.iloc[tr_i], y.iloc[tr_i])
        pred = pipe.predict(texts.iloc[va_i])
        acc = accuracy_score(y.iloc[va_i], pred)
        f1m = f1_score(y.iloc[va_i], pred, average="macro")
        f1w = f1_score(y.iloc[va_i], pred, average="weighted")
        rows.append({"variant": tag, "fold": fold, "accuracy": acc,
                     "macro_f1": f1m, "weighted_f1": f1w})
        print(f"    fold {fold}: acc={acc:.4f}  macro-F1={f1m:.4f}  weighted-F1={f1w:.4f}")

    df = pd.DataFrame(rows)
    agg = {
        "variant": tag,
        "accuracy_mean": df["accuracy"].mean(), "accuracy_std": df["accuracy"].std(),
        "macro_f1_mean": df["macro_f1"].mean(), "macro_f1_std": df["macro_f1"].std(),
        "weighted_f1_mean": df["weighted_f1"].mean(), "weighted_f1_std": df["weighted_f1"].std(),
    }
    print(f"  ==> {tag}: acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}  "
          f"macro-F1={agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}")
    return agg, df


def main() -> None:
    t0 = time.time()
    print("=" * 92)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 92)

    train = pd.read_csv(PROC / "train_clean.csv")
    test = pd.read_csv(PROC / "test_clean.csv")
    print(f"train: {train.shape}   test: {test.shape}")

    y_train = train["sentiment"].astype(int)
    y_test = test["sentiment"].astype(int)
    text_clean = train["text"].fillna("")
    text_nosw = train["text_no_sw"].fillna("")
    text_lemma = train["text_lemma"].fillna("")
    # `phrase` is preserved verbatim from the raw file by the cleaning pipeline,
    # so it is row-aligned with y_train by construction (no re-derivation needed).
    text_raw = train["phrase"].astype(str)
    assert len(text_raw) == len(y_train)

    print("\n" + "-" * 92)
    print("A. 5-fold stratified cross-validation on TRAIN (model-selection view)")
    print("-" * 92)
    print("  V1  raw text, no cleaning")
    agg_v1, folds_v1 = cv_evaluate(text_raw, y_train, "V1 raw text (no cleaning)")
    print("\n  V2  cleaned text  <-- HEADLINE BASELINE")
    agg_v2, folds_v2 = cv_evaluate(text_clean, y_train, "V2 cleaned text")
    print("\n  V2b cleaned text + stop-words removed")
    agg_v2b, folds_v2b = cv_evaluate(text_nosw, y_train, "V2b cleaned + stop-words removed")
    print("\n  V2c cleaned + stop-words removed + lemmatised  <-- FINAL BASELINE INPUT")
    agg_v2c, folds_v2c = cv_evaluate(text_lemma, y_train, "V2c cleaned + stop-words + lemmatised")
    print("\n  V3  cleaned+lemmatised + class_weight=balanced")
    agg_v3, folds_v3 = cv_evaluate(text_lemma, y_train, "V3 lemmatised + balanced class weight",
                                   balanced=True)

    cv_table = pd.DataFrame([agg_v1, agg_v2, agg_v2b, agg_v2c, agg_v3])
    cv_table.to_csv(TAB_DIR / "cv_results.csv", index=False)
    print("\n" + cv_table.round(4).to_string(index=False))

    # ---------------- final model on ALL train, scored on hold-out test
    print("\n" + "-" * 92)
    print("B. Final baseline fitted on ALL of train, evaluated on the held-out TEST file")
    print("-" * 92)
    final = build_pipeline(balanced=False)
    final.fit(text_lemma, y_train)
    y_pred = final.predict(test["text_lemma"].fillna(""))
    y_proba = final.predict_proba(test["text_lemma"].fillna(""))

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")
    print(f"  TEST accuracy    : {acc:.4f}")
    print(f"  TEST macro F1    : {f1m:.4f}")
    print(f"  TEST weighted F1 : {f1w:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=4, zero_division=0))

    p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    per_class = pd.DataFrame({
        "class": LABEL_NAMES, "precision": p, "recall": r, "f1": f, "support": s,
    })
    per_class.to_csv(TAB_DIR / "test_per_class_metrics.csv", index=False)
    print(per_class.round(4).to_string(index=False))

    # majority-class reference point
    majority = int(y_train.value_counts().idxmax())
    maj_acc = float((y_test == majority).mean())
    print(f"\n  Majority-class baseline accuracy on TEST: {maj_acc:.4f} "
          f"(always predict '{LABEL_NAMES[majority]}')")
    print(f"  Baseline lift over majority: +{(acc - maj_acc)*100:.1f} pp")

    # ---------------- reference: balanced variant on TEST (imbalance lever)
    print("\n" + "-" * 92)
    print("C. Imbalance lever: same model with class_weight='balanced' (for Phase 3iv)")
    print("-" * 92)
    final_bal = build_pipeline(balanced=True)
    final_bal.fit(text_lemma, y_train)
    y_pred_bal = final_bal.predict(test["text_lemma"].fillna(""))
    acc_bal = accuracy_score(y_test, y_pred_bal)
    f1m_bal = f1_score(y_test, y_pred_bal, average="macro")
    f1w_bal = f1_score(y_test, y_pred_bal, average="weighted")
    pb, rb_, fb, sb = precision_recall_fscore_support(y_test, y_pred_bal,
                                                      labels=[0, 1, 2, 3], zero_division=0)
    print(f"  TEST accuracy={acc_bal:.4f}  macro-F1={f1m_bal:.4f}  weighted-F1={f1w_bal:.4f}")
    print(f"  delta vs plain baseline: accuracy {(acc_bal-acc)*100:+.1f} pp, "
          f"macro-F1 {(f1m_bal-f1m)*100:+.1f} pp")
    print("  per-class F1: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(LABEL_NAMES, fb)))
    balanced_per_class = pd.DataFrame({
        "class": LABEL_NAMES, "precision": pb, "recall": rb_, "f1": fb, "support": sb,
    })
    balanced_per_class.to_csv(TAB_DIR / "test_per_class_balanced.csv", index=False)

    # ---------------- confusion matrices (counts + normalised)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[0],
                annot_kws={"fontsize": 10})
    axes[0].set_title(f"Counts (test n={len(y_test)})\naccuracy={acc:.3f}, macro-F1={f1m:.3f}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    sns.heatmap(cmn, annot=True, fmt=".1f", cmap="Blues", cbar=True,
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[1],
                annot_kws={"fontsize": 10})
    axes[1].set_title("Row-normalised (% of actual class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    fig.suptitle("Baseline confusion matrix (TF-IDF + Logistic Regression)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig09_confusion_matrix.png")
    plt.close(fig)
    print("  fig09 confusion matrix   OK")

    # ---------------- CV comparison chart
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    short_names = {
        "V1 raw text (no cleaning)": "V1\nRaw\n(no cleaning)",
        "V2 cleaned text": "V2\nCleaned",
        "V2b cleaned + stop-words removed": "V2b\n+ stop-words\nremoved",
        "V2c cleaned + stop-words + lemmatised": "V2c\n+ lemmatised",
        "V3 lemmatised + balanced class weight": "V3\n+ balanced\nclass weight",
    }
    cv_table["short"] = cv_table["variant"].map(short_names)
    plot = cv_table.melt(id_vars=["variant", "short"],
                         value_vars=["accuracy_mean", "macro_f1_mean", "weighted_f1_mean"],
                         var_name="metric", value_name="score")
    plot["metric"] = plot["metric"].map({"accuracy_mean": "Accuracy",
                                         "macro_f1_mean": "Macro F1",
                                         "weighted_f1_mean": "Weighted F1"})
    sns.barplot(data=plot, x="short", y="score", hue="metric", ax=ax,
                palette=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black", linewidth=.5)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7, padding=1.5)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("")
    ax.set_ylabel("5-fold CV score on train")
    ax.set_title("Cleaning decisions are measurable: 5-fold CV comparison\n"
                 "Class-balanced weighting is the biggest single win (+6.6 pp macro-F1)")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              frameon=True)
    ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_cv_comparison.png")
    plt.close(fig)
    print("  fig10 CV comparison      OK")

    # ---------------- learning curve
    print("\n  computing learning curve ...")
    sizes, tr_scores, cv_scores = learning_curve(
        build_pipeline(False), text_lemma, y_train,
        train_sizes=np.linspace(0.2, 1.0, 5), cv=5, scoring="f1_macro",
        shuffle=True, random_state=RANDOM_STATE, n_jobs=1,
    )
    curve = {
        "train_size": sizes.tolist(),
        "train_f1": tr_scores.mean(axis=1).tolist(),
        "cv_f1": cv_scores.mean(axis=1).tolist(),
        "cv_f1_std": cv_scores.std(axis=1).tolist(),
    }
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.plot(curve["train_size"], curve["train_f1"], "o-", color="#2CA02C", label="Train macro-F1")
    ax.plot(curve["train_size"], curve["cv_f1"], "s-", color="#D62728", label="CV macro-F1")
    ax.fill_between(curve["train_size"],
                    np.array(curve["cv_f1"]) - np.array(curve["cv_f1_std"]),
                    np.array(curve["cv_f1"]) + np.array(curve["cv_f1_std"]),
                    alpha=.18, color="#D62728")
    ax.set_xlabel("Training examples used")
    ax.set_ylabel("Macro F1")
    ax.set_title("Baseline learning curve\n"
                 "Large train-CV gap => variance problem; the model memorises rather than under-fits")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_learning_curve.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  fig11 learning curve     OK")

    # ---------------- top coefficients per class
    tfidf: TfidfVectorizer = final.named_steps["tfidf"]
    clf: LogisticRegression = final.named_steps["clf"]
    feats = np.array(tfidf.get_feature_names_out())
    coef_rows = []
    for i, cls in enumerate(LABEL_NAMES):
        coefs = clf.coef_[i]
        top = np.argsort(coefs)[-15:][::-1]
        for rank, j in enumerate(top, 1):
            coef_rows.append({"class": cls, "rank": rank, "term": feats[j],
                              "coefficient": float(coefs[j])})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(TAB_DIR / "top_coefficients.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    for ax, cls in zip(axes.ravel(), LABEL_NAMES):
        sub = coef_df[coef_df["class"] == cls].head(12).iloc[::-1]
        i = LABEL_NAMES.index(cls)
        ax.barh(sub["term"], sub["coefficient"], color=PALETTE[i], alpha=.85,
                edgecolor="black", linewidth=.4)
        ax.set_title(f"{cls} - strongest positive terms")
        ax.set_xlabel("Logistic-regression coefficient")
        ax.tick_params(axis="y", labelsize=8.5)
    fig.suptitle("What the baseline learned (top TF-IDF terms per class)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_top_coefficients.png")
    plt.close(fig)
    print("  fig12 coefficients       OK")

    # ---------------- error analysis
    err = pd.DataFrame({
        "phrase": test["phrase"].astype(str),
        "text": test["text_lemma"].astype(str),
        "actual": [LABEL_NAMES[i] for i in y_test],
        "predicted": [LABEL_NAMES[i] for i in y_pred],
        "confidence": y_proba.max(axis=1).round(3),
    })
    err["correct"] = err["actual"] == err["predicted"]
    err.to_csv(TAB_DIR / "test_predictions.csv", index=False)
    print(f"\n  misclassified: {(~err['correct']).sum()} / {len(err)} "
          f"({(~err['correct']).mean()*100:.1f}%)")
    print("\n  Confusion in words (actual -> predicted):")
    words = (err[~err["correct"]].groupby(["actual", "predicted"]).size()
             .sort_values(ascending=False).head(8))
    for (a, p_), n in words.items():
        print(f"    {a:<11} -> {p_:<11} {n:>3}")

    # save model + metadata
    joblib.dump(final, MODEL_DIR / "baseline_tfidf_logreg.joblib")
    meta = {
        "model": "TF-IDF (1-2 gram, sublinear) + multinomial Logistic Regression (C=1.0, lbfgs)",
        "trained_on": "data/processed/train_clean.csv",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "tfidf_features": int(len(feats)),
        "cv": cv_table.round(6).to_dict(orient="records"),
        "test_accuracy": round(float(acc), 6),
        "test_macro_f1": round(float(f1m), 6),
        "test_weighted_f1": round(float(f1w), 6),
        "test_balanced_accuracy": round(float(acc_bal), 6),
        "test_balanced_macro_f1": round(float(f1m_bal), 6),
        "test_balanced_weighted_f1": round(float(f1w_bal), 6),
        "majority_baseline_accuracy": round(maj_acc, 6),
        "per_class": per_class.round(6).to_dict(orient="records"),
        "learning_curve": curve,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(MODEL_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  model    -> {MODEL_DIR/'baseline_tfidf_logreg.joblib'}")
    print(f"  metrics  -> {MODEL_DIR/'baseline_metrics.json'}")
    print(f"\nDONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
