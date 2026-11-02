"""
Transformer probe: how far does a pre-trained encoder get past the ~50% TF-IDF ceiling?

Fine-tunes a small BERT-family encoder on TRAIN and reports held-out TEST accuracy.
This is an exploration script (test is watched); production selection lives in
`train_final.py`.

Run: python src/explore_transformer.py [model_name] [epochs]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "distilbert-base-uncased"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(16)

train = pd.read_csv(PROC / "train_clean.csv")
test = pd.read_csv(PROC / "test_clean.csv")
LABELS = ["Negative", "Positive", "Irrelevant", "Neutral"]


class TweetDS(Dataset):
    def __init__(self, texts, labels, tok, max_len=64):
        self.enc = tok(list(texts), truncation=True, max_length=max_len, padding=False)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def collate(batch, tok):
    labels = torch.tensor([b.pop("labels") for b in batch])
    out = tok.pad(batch, padding=True, return_tensors="pt")
    out["labels"] = labels
    return out


@torch.no_grad()
def predict(model, texts, tok, bs=64):
    model.eval()
    loader = DataLoader(TweetDS(texts, [0] * len(texts), tok), batch_size=bs,
                        collate_fn=lambda b: collate(b, tok))
    logits = []
    for batch in loader:
        batch.pop("labels")
        logits.append(model(**batch).logits)
    return torch.cat(logits).softmax(-1).numpy()


def train_one(texts, labels, texts_val=None, labels_val=None, epochs=EPOCHS, lr=3e-5):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=4, ignore_mismatched_sizes=True)
    ds = TweetDS(texts, labels, tok)
    loader = DataLoader(ds, batch_size=16, shuffle=True,
                        collate_fn=lambda b: collate(b, tok))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
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
        msg = f"    epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}"
        if texts_val is not None:
            pa = predict(model, texts_val, tok).argmax(1)
            msg += f"  val_acc={accuracy_score(labels_val, pa):.4f}"
        print(msg, flush=True)
    return model, tok


t0 = time.time()
print(f"MODEL={MODEL}  epochs={EPOCHS}")
col = "phrase"
Xtr, Xte = train[col].astype(str).tolist(), test[col].astype(str).tolist()
ytr, yte = train["sentiment"].astype(int).tolist(), test["sentiment"].astype(int).tolist()

# quick 3-fold CV for an honest selection signal, then full-train fit for the test score
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
cv_accs, cv_f1s = [], []
for k, (tr_i, va_i) in enumerate(skf.split(Xtr, ytr), 1):
    print(f"  fold {k}")
    m, tok = train_one([Xtr[i] for i in tr_i], [ytr[i] for i in tr_i],
                       [Xtr[i] for i in va_i], [ytr[i] for i in va_i], epochs=EPOCHS)
    p = predict(m, [Xtr[i] for i in va_i], tok).argmax(1)
    a = accuracy_score([ytr[i] for i in va_i], p)
    f = f1_score([ytr[i] for i in va_i], p, average="macro")
    cv_accs.append(a); cv_f1s.append(f)
    print(f"  ==> fold {k}: acc={a:.4f} macroF1={f:.4f}", flush=True)

print(f"CV acc={np.mean(cv_accs):.4f}±{np.std(cv_accs):.4f}  macroF1={np.mean(cv_f1s):.4f}")

m, tok = train_one(Xtr, ytr, epochs=EPOCHS)
proba = predict(m, Xte, tok)
p = proba.argmax(1)
acc = accuracy_score(yte, p)
f1m = f1_score(yte, p, average="macro")
print(f"TEST acc={acc:.4f}  macroF1={f1m:.4f}")
print(classification_report(yte, p, target_names=LABELS, digits=4))

tag = MODEL.replace("/", "_")
pd.DataFrame({"phrase": test["phrase"], "actual": [LABELS[i] for i in yte],
              "predicted": [LABELS[i] for i in p],
              "confidence": proba.max(1).round(4)}).to_csv(
    OUT / f"_transformer_predictions_{tag}.csv", index=False)
json.dump({"model": MODEL, "epochs": EPOCHS, "cv_acc": float(np.mean(cv_accs)),
           "cv_acc_sd": float(np.std(cv_accs)), "cv_f1": float(np.mean(cv_f1s)),
           "test_acc": float(acc), "test_f1": float(f1m),
           "runtime_s": round(time.time() - t0, 1)},
          open(OUT / f"_transformer_{tag}.json", "w"), indent=2)
print(f"done in {time.time()-t0:.0f}s")
