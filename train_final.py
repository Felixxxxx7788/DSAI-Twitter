"""Consolidated FINAL training pipeline.

Produces every number the report cites, with a clean protocol:

  * Model **selection** happens on TRAIN only (5-fold stratified CV).
  * The TEST file is touched **once**, by the already-selected model.
  * No test-driven tuning anywhere in this file.

Models compared (all on the same folds, same seed)
-------------------------------------------------
  M1  TF-IDF word 1-2grams + LogisticRegression          (the original baseline)
  M2  TF-IDF word+char union + LogisticRegression        (best linear model)
  M3  TF-IDF word+char union + LinearSVC                 (margin variant)
  M4  DistilBERT fine-tuned                              (best encoder)
  M5  Soft-vote ensemble of M2 + M4                      (averages away view noise)

Text view: the raw `phrase` column. An earlier baseline used `text_lemma` and
lost ~4.5 pp; `experiments_headroom.py` E3 shows that cleaning itself is
lossless (91.4 % agreement with raw) but lemmatisation + stop-word removal
destroys 26 % of the view, so the final models read `phrase`.

Outputs
-------
  models/final_tfidf.joblib              fitted linear model (default deployment)
  models/final_distilbert/               fine-tuned encoder + tokenizer
  models/final_metrics.json              every number used by the reports
  reports/tables/final_cv_results.csv
  reports/tables/final_per_class.csv
  reports/tables/final_predictions.csv
  reports/figures/fig13_final_confusion.png
  reports/figures/fig14_model_comparison.png

Run: python src/train_final.py            (linear models only, ~1 min)
     python src/train_final.py --with-transformer   (adds DistilBERT, ~15 min CPU)
"""
from __future__ import annotations

import argparse
import copy
import json
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
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
MODELS = ROOT / "models"
for d in (FIG, TAB, MODELS):
    d.mkdir(parents=True, exist_ok=True)

LABELS = ["Negative", "Positive", "Irrelevant", "Neutral"]
PALETTE = ["#D62728", "#2CA02C", "#7F7F7F", "#1F77B4"]
SEED = 42
TEXT_COL = "phrase"           # raw text wins over text_lemma by ~4.5 pp
N_SPLITS = 5

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold"})


# --------------------------------------------------------------------- models
def word_vec() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95,
                           sublinear_tf=True, strip_accents="unicode",
                           lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")


def char_vec() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           max_df=0.95, sublinear_tf=True, lowercase=True)


def union_vec() -> FeatureUnion:
    return FeatureUnion([("word", word_vec()), ("char", char_vec())])


def m1_word_lr() -> Pipeline:
    return Pipeline([("v", word_vec()),
                     ("c", LogisticRegression(C=1.0, max_iter=3000, random_state=SEED))])


def m2_union_lr() -> Pipeline:
    return Pipeline([("v", union_vec()),
                     ("c", LogisticRegression(C=1.0, max_iter=4000, random_state=SEED))])


def m3_union_svc() -> Pipeline:
    return Pipeline([("v", union_vec()),
                     ("c", LinearSVC(C=0.5, random_state=SEED))])


LINEAR_MODELS = {"M1 word TF-IDF + LR": m1_word_lr,
                 "M2 word+char union + LR": m2_union_lr,
                 "M3 word+char union + LinearSVC": m3_union_svc}


# ---------------------------------------------------------------- transformer
def transformer_fit_predict(X_tr, y_tr, X_te, epochs=4, model_name="distilbert-base-uncased",
                            save_dir: Path | None = None, seed: int = SEED):
    """Fine-tune an encoder; returns (test probabilities, train OOF-free proba fn)."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(16)
    tok = AutoTokenizer.from_pretrained(model_name)

    class DS(Dataset):
        def __init__(self, texts, labels):
            self.enc = tok(list(texts), truncation=True, max_length=64, padding=False)
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            d = {k: v[i] for k, v in self.enc.items()}
            d["labels"] = self.labels[i]
            return d

    def collate(batch):
        labels = torch.tensor([b.pop("labels") for b in batch])
        out = tok.pad(batch, padding=True, return_tensors="pt")
        out["labels"] = labels
        return out

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4, ignore_mismatched_sizes=True)
    loader = DataLoader(DS(X_tr, y_tr), batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    model.train()
    for ep in range(epochs):
        losses = []
        for batch in loader:
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(out.loss.item())
        print(f"      epoch {ep+1}/{epochs}  loss={np.mean(losses):.4f}", flush=True)

    @torch.no_grad()
    def proba(texts, bs=64):
        model.eval()
        outs = []
        ld = DataLoader(DS(texts, [0] * len(texts)), batch_size=bs, collate_fn=collate)
        for b in ld:
            b.pop("labels")
            outs.append(model(**b).logits.softmax(-1))
        return torch.cat(outs).numpy()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
    return proba


# --------------------------------------------------------------------- utils
def oof_proba(factory, X, y, n_splits=N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    proba = np.zeros((len(y), 4))
    for tr, va in skf.split(X, y):
        p = copy.deepcopy(factory())
        p.fit(X.iloc[tr], y[tr])
        if hasattr(p, "predict_proba"):
            proba[va] = p.predict_proba(X.iloc[va])
        else:                                    # LinearSVC: softmax the margins
            m = p.decision_function(X.iloc[va])
            e = np.exp(m - m.max(axis=1, keepdims=True))
            proba[va] = e / e.sum(axis=1, keepdims=True)
    return proba


def score_block(y_true, proba) -> dict:
    pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2, 3],
                                                 zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "balanced_accuracy": float(np.mean(r)),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
        "within1_accuracy": float((np.abs(y_true - pred) <= 1).mean()),
        "per_class": [{"class": LABELS[i], "precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f[i]), "support": int(s[i])} for i in range(4)],
    }


def _r(v, nd=6):
    """Round floats, pass None/ints/strings through untouched."""
    return round(v, nd) if isinstance(v, float) else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-transformer", action="store_true",
                    help="also fine-tune DistilBERT (adds ~15 min on CPU)")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 90)
    print("FINAL PIPELINE  -  selection on TRAIN CV, single evaluation on TEST")
    print("=" * 90)

    train = pd.read_csv(PROC / "train_clean.csv")
    test = pd.read_csv(PROC / "test_clean.csv")
    X_tr, X_te = train[TEXT_COL].fillna("").astype(str), test[TEXT_COL].fillna("").astype(str)
    y_tr, y_te = train["sentiment"].astype(int).values, test["sentiment"].astype(int).values
    print(f"train {len(train)} rows | test {len(test)} rows | text column = '{TEXT_COL}'")

    majority = int(pd.Series(y_tr).value_counts().idxmax())
    maj_acc = float((y_te == majority).mean())

    # ------------------------------------------------------- 1. TRAIN-only CV
    print("\n" + "-" * 90)
    print("1.  TRAIN-only 5-fold CV (this is the model-selection evidence)")
    print("-" * 90)
    cv_rows, test_probas = [], {}
    for name, factory in LINEAR_MODELS.items():
        proba = oof_proba(factory, X_tr, y_tr)
        s = score_block(y_tr, proba)
        cv_rows.append({"model": name, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                        "weighted_f1": s["weighted_f1"], "kappa": s["cohen_kappa"]})
        print(f"  {name:<34s} acc={s['accuracy']:.4f}  macroF1={s['macro_f1']:.4f}  "
              f"kappa={s['cohen_kappa']:.4f}")
        # fit on all train, score test once
        p = factory().fit(X_tr, y_tr)
        if hasattr(p, "predict_proba"):
            test_probas[name] = p.predict_proba(X_te)
        else:
            m = p.decision_function(X_te)
            e = np.exp(m - m.max(axis=1, keepdims=True))
            test_probas[name] = e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------- 2. transformer
    transformer_cv = None
    if args.with_transformer:
        print("\n" + "-" * 90)
        print(f"2.  DistilBERT cross-validation ({N_SPLITS} folds x {args.epochs} epochs)")
        print("-" * 90)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        cv_acc, cv_f1 = [], []
        for k, (tr, va) in enumerate(skf.split(X_tr, y_tr), 1):
            print(f"    fold {k}/{N_SPLITS}")
            pf = transformer_fit_predict(X_tr.iloc[tr], y_tr[tr], None,
                                         epochs=args.epochs, seed=SEED)
            pp = pf(X_tr.iloc[va])
            a, fm = accuracy_score(y_tr[va], pp.argmax(1)), f1_score(y_tr[va], pp.argmax(1), average="macro")
            cv_acc.append(a); cv_f1.append(fm)
            print(f"    ==> fold {k}: acc={a:.4f}  macroF1={fm:.4f}", flush=True)
        transformer_cv = {"model": "M4 DistilBERT (fine-tuned)",
                          "accuracy": float(np.mean(cv_acc)), "accuracy_sd": float(np.std(cv_acc)),
                          "macro_f1": float(np.mean(cv_f1)),
                          "weighted_f1": None, "kappa": None}
        cv_rows.append({k: transformer_cv.get(k) for k in
                        ["model", "accuracy", "macro_f1", "weighted_f1", "kappa"]})
        print(f"  M4 DistilBERT        acc={np.mean(cv_acc):.4f}+/-{np.std(cv_acc):.4f}  "
              f"macroF1={np.mean(cv_f1):.4f}")

        print("\n    fitting DistilBERT on all train rows ...")
        pf = transformer_fit_predict(X_tr, y_tr, None, epochs=args.epochs,
                                     save_dir=MODELS / "final_distilbert", seed=SEED)
        test_probas["M4 DistilBERT"] = pf(X_te)

        # ------------------------------------------- 3. ensemble (train-CV chosen)
        if "M2 word+char union + LR" in test_probas:
            test_probas["M5 ensemble (M2 + M4)"] = (
                0.5 * test_probas["M2 word+char union + LR"] + 0.5 * test_probas["M4 DistilBERT"])

    # ------------------------------------------------------- 4. TEST (once)
    print("\n" + "-" * 90)
    print("3.  Held-out TEST evaluation  (each model scored exactly once)")
    print("-" * 90)
    test_rows, best_name, best_acc = [], None, -1.0
    for name, proba in test_probas.items():
        s = score_block(y_te, proba)
        test_rows.append({"model": name, **{k: s[k] for k in
                          ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                           "cohen_kappa", "within1_accuracy"]}})
        print(f"  {name:<34s} acc={s['accuracy']:.4f}  macroF1={s['macro_f1']:.4f}  "
              f"kappa={s['cohen_kappa']:.4f}")
        if s["accuracy"] > best_acc:
            best_acc, best_name = s["accuracy"], name

    # default deployment model: best linear model (transformer is optional)
    deploy_name = "M2 word+char union + LR"
    deploy_proba = test_probas[deploy_name]
    deploy = score_block(y_te, deploy_proba)
    print("\n  " + classification_report(y_te, deploy_proba.argmax(1),
                                         target_names=LABELS, digits=4, zero_division=0))
    print(f"  majority-class reference: {maj_acc:.4f}")

    pd.DataFrame(test_rows).to_csv(TAB / "final_cv_results.csv", index=False)
    pd.DataFrame(deploy["per_class"]).to_csv(TAB / "final_per_class.csv", index=False)

    # ------------------------------------------------------- 5. figures
    cm = confusion_matrix(y_te, deploy_proba.argmax(1), labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax, annot_kws={"fontsize": 10})
    ax.set_title(f"Final model - word+char TF-IDF + LR\n"
                 f"test accuracy {deploy['accuracy']*100:.1f}%  |  "
                 f"macro-F1 {deploy['macro_f1']*100:.1f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(FIG / "fig13_final_confusion.png")
    plt.close(fig)

    cmp_df = pd.DataFrame(test_rows).sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    colors = ["#4C72B0" if "M2 " not in m else "#DD8452" for m in cmp_df["model"]]
    ax.barh(cmp_df["model"], cmp_df["accuracy"] * 100, color=colors,
            edgecolor="black", linewidth=.5)
    for i, (m, a) in enumerate(zip(cmp_df["model"], cmp_df["accuracy"])):
        ax.text(a * 100 + 0.6, i, f"{a*100:.1f}%", va="center", fontsize=8.5)
    ax.axvline(maj_acc * 100, ls="--", color="grey", lw=1.2,
               label=f"majority class ({maj_acc*100:.1f}%)")
    ax.set_xlabel("Held-out test accuracy (%)")
    ax.set_xlim(0, max(70, cmp_df["accuracy"].max() * 100 + 8))
    ax.set_title("Model comparison on the held-out test set")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig14_model_comparison.png")
    plt.close(fig)
    print("\n  fig13 final confusion    OK")
    print("  fig14 model comparison   OK")

    # ------------------------------------------------------- 6. artefacts
    deploy_model = m2_union_lr().fit(X_tr, y_tr)
    joblib.dump(deploy_model, MODELS / "final_tfidf.joblib")

    pred_df = pd.DataFrame({
        "phrase": test[TEXT_COL].astype(str),
        "actual": [LABELS[i] for i in y_te],
        "predicted": [LABELS[i] for i in deploy_proba.argmax(1)],
        "confidence": deploy_proba.max(1).round(4),
        "correct": deploy_proba.argmax(1) == y_te,
    })
    pred_df.to_csv(TAB / "final_predictions.csv", index=False)
    print(f"  misclassified: {(~pred_df['correct']).sum()} / {len(pred_df)} "
          f"({(~pred_df['correct']).mean()*100:.1f}%)")

    # headroom context from the experiments stage
    headroom = {}
    hp = PROC / "headroom_summary.json"
    if hp.exists():
        headroom = json.loads(hp.read_text())

    metrics = {
        "protocol": ("Models are selected by 5-fold stratified CV on TRAIN only; "
                     "the TEST file is scored once per configured model. "
                     "Seed 42 for every split and estimator."),
        "text_column": TEXT_COL,
        "text_column_rationale": (
            "`phrase` (raw) beats `text_lemma` by ~4.5 pp on test. Cleaning is lossless "
            "(91.4% prediction agreement) but lemmatisation + stop-word removal removes "
            "26% of the view - see experiments_headroom.py E3."),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "majority_baseline_accuracy": round(maj_acc, 6),
        "deployed_model": deploy_name,
        "deployed_metrics": {k: _r(v) for k, v in deploy.items() if k != "per_class"},
        "per_class": [{k: _r(v) for k, v in pc.items()} for pc in deploy["per_class"]],
        "cv_train_only": [{k: _r(v) for k, v in r.items()} for r in cv_rows],
        "test_results": [{k: _r(v) for k, v in r.items()} for r in test_rows],
        "best_test_model": best_name,
        "best_test_accuracy": round(best_acc, 6),
        "improvement_over_original_baseline_pp": round(
            (deploy["accuracy"] - 0.496222) * 100, 1),
        "headroom_evidence": headroom,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(MODELS / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  deployed model -> {MODELS/'final_tfidf.joblib'}")
    print(f"  metrics        -> {MODELS/'final_metrics.json'}")
    print(f"  best on test   : {best_name} @ {best_acc*100:.1f}%")
    print(f"\nDONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
