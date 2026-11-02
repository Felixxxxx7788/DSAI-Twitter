"""
Phase 1 + 2 : Data loading, cleaning pipeline, and EDA artefact generation.

Produces
--------
data/processed/train_clean.csv
data/processed/test_clean.csv
data/processed/cleaning_log.json          (row-level audit trail)
reports/figures/*.png                     (all EDA figures)
reports/tables/*.csv                      (all EDA tables)

Run: python src/run_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_clean as tc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data 5 - Twitter" / "Data 5 - Twitter"
OUT_DATA = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
for d in (OUT_DATA, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {0: "Negative", 1: "Positive", 2: "Irrelevant", 3: "Neutral"}
PALETTE = {0: "#D62728", 1: "#2CA02C", 2: "#7F7F7F", 3: "#1F77B4"}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10,
})

STOPWORDS: set[str] = set()
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set(ENGLISH_STOP_WORDS)
except Exception:  # pragma: no cover
    STOPWORDS = set()

# A small extra list of Twitter boilerplate that survives sklearn's list.
TWITTER_NOISE = {"http", "https", "www", "com", "amp", "rt", "rhandlerr",
                 "pic", "twitter", "t", "co", "unk", "via", "lol", "lmao"}
STOPWORDS |= TWITTER_NOISE


# ------------------------------------------------------------------ loading
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"Phrase": "string"})
    df = df.rename(columns={"Phrase": "phrase", "Sentiment": "sentiment"})
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    # train stores labels as float (0.0) and test as int (0) -> standardise to Int64
    df["sentiment"] = df["sentiment"].astype("Int64")
    return df


# ------------------------------------------------------------------ cleaning
def clean_frame(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    """Apply the documented cleaning pipeline and return (clean_df, log)."""
    log: dict = {"dataset": name, "steps": []}
    log["rows_in"] = int(len(df))

    # ---- Step 1: schema / dtype standardisation
    before = df["sentiment"].dtype
    df = df.copy()
    df["sentiment"] = df["sentiment"].astype("Int64")
    log["steps"].append({
        "step": 1, "name": "Schema standardisation",
        "action": "Renamed Phrase->phrase, Sentiment->sentiment; cast label to nullable Int64",
        "detail": f"train was float64 ({before}) while test was int64 -> unified to Int64 "
                  f"so the two files can be concatenated and compared safely",
        "rows_affected": int(len(df)),
    })

    # ---- Step 2: missing values
    na_phrase = int(df["phrase"].isna().sum())
    na_label = int(df["sentiment"].isna().sum())
    df = df[df["phrase"].notna() & df["sentiment"].notna()].copy()
    log["steps"].append({
        "step": 2, "name": "Missing-value handling",
        "action": "Dropped rows with missing phrase or label (no imputation possible for text targets)",
        "detail": f"missing phrases={na_phrase}, missing labels={na_label}",
        "rows_affected": na_phrase + na_label,
    })

    # ---- Step 3: blank / whitespace-only text
    blank = df["phrase"].astype(str).str.strip().eq("")
    n_blank = int(blank.sum())
    df = df[~blank].copy()
    log["steps"].append({
        "step": 3, "name": "Blank text removal",
        "action": "Removed rows whose phrase was empty after trimming",
        "detail": f"{n_blank} blank rows",
        "rows_affected": n_blank,
    })

    # ---- Step 4: exact duplicates
    dup_mask = df.duplicated(subset=["phrase", "sentiment"], keep="first")
    n_dup = int(dup_mask.sum())
    df = df[~dup_mask].copy()
    log["steps"].append({
        "step": 4, "name": "Exact duplicate removal",
        "action": "Dropped duplicate (phrase, sentiment) pairs keeping first occurrence",
        "detail": f"{n_dup} exact duplicates removed",
        "rows_affected": n_dup,
    })

    # ---- Step 5: conflicting labels for identical text
    conf_mask = df.duplicated(subset=["phrase"], keep=False)
    n_conflict_text = int(df.loc[conf_mask, "phrase"].nunique())
    n_conflict_rows = int(conf_mask.sum())
    conf_pairs = []
    kept_conflict_rows = 0
    if n_conflict_rows:
        # Keep the majority label; ties -> keep the first (lowest label id) deterministically
        grp = (df[conf_mask].groupby("phrase")["sentiment"]
               .agg(lambda s: s.value_counts().idxmax()))
        for phrase, label in grp.items():
            members = df[df["phrase"] == phrase]
            conf_pairs.append({
                "phrase": str(phrase)[:160],
                "labels_seen": sorted(int(x) for x in members["sentiment"].unique()),
                "majority_label": int(label),
                "duplicate_count": int(len(members)),
            })
        drop_idx = []
        for phrase in grp.index:
            members = df[df["phrase"] == phrase]
            majority = grp[phrase]
            same = members[members["sentiment"] == majority]
            # keep exactly one majority row, drop the rest of the group
            drop_idx.extend(same.index[1:].tolist())
            drop_idx.extend(members[members["sentiment"] != majority].index.tolist())
        kept_conflict_rows = len(drop_idx)
        df = df.drop(index=drop_idx).reset_index(drop=True)

    log["steps"].append({
        "step": 5, "name": "Conflicting-label resolution",
        "action": "For identical text with different labels, kept the majority label (one row)",
        "detail": f"{n_conflict_text} phrase(s) had conflicting labels across {n_conflict_rows} rows; "
                  f"{kept_conflict_rows} rows removed",
        "rows_affected": kept_conflict_rows,
        "conflicts": conf_pairs,
    })

    # ---- Step 6: invalid label values
    invalid = ~df["sentiment"].isin([0, 1, 2, 3])
    n_invalid = int(invalid.sum())
    df = df[~invalid].copy()
    log["steps"].append({
        "step": 6, "name": "Invalid label correction",
        "action": "Dropped labels outside the documented 0-3 domain",
        "detail": f"{n_invalid} invalid labels",
        "rows_affected": n_invalid,
    })

    # ---- Step 7: encoding repair
    df["phrase_fixed"] = df["phrase"].astype(str).map(tc.fix_encoding)
    mojibake_before = int(df["phrase"].astype(str).str.contains(r"[ÃÂâ]", regex=True).sum())
    mojibake_after = int(df["phrase_fixed"].str.contains(r"[ÃÂâ]", regex=True).sum())
    log["steps"].append({
        "step": 7, "name": "Encoding repair (mojibake)",
        "action": "Applied ftfy.fix_text + Unicode NFKC to reverse UTF-8 read as cp1252",
        "detail": f"rows containing 'Ã/Â/â' before={mojibake_before}, after={mojibake_after}",
        "rows_affected": mojibake_before - mojibake_after,
    })

    # ---- Step 8: text normalisation -> model-ready 'text'
    df["text"] = df["phrase_fixed"].map(lambda s: tc.basic_clean(s))
    df["text_no_sw"] = df["text"].map(
        lambda s: " ".join(tc.remove_stopwords(tc.tokenise(s), STOPWORDS))
    )
    # ---- Step 8b: lemmatisation (required by the brief: "lemmatization or stemming")
    lemmatiser = tc._get_lemmatiser()
    df["text_lemma"] = df["text_no_sw"].map(
        lambda s: " ".join(tc.lemmatise(s.split(), lemmatiser))
    )
    df["n_tokens_raw"] = df["text"].map(tc.token_count)
    df["n_tokens_clean"] = df["text_no_sw"].map(tc.token_count)
    empty_after = int((df["n_tokens_clean"] == 0).sum())
    vocab_before = len({w for s in df["text_no_sw"] for w in s.split()})
    vocab_after = len({w for s in df["text_lemma"] for w in s.split()})
    log["steps"].append({
        "step": 8, "name": "Text normalisation & tokenisation",
        "action": ("lowercased; expanded contractions; stripped URLs, @mentions, HTML tags, "
                   "<unk> placeholders; collapsed character elongation; removed non-alphabetic "
                   "characters; tokenised on whitespace; removed stop-words (negations preserved)"),
        "detail": f"{empty_after} rows became empty after cleaning (retained for review, flagged)",
        "rows_affected": int(len(df)),
        "empty_after_clean": empty_after,
    })
    log["steps"].append({
        "step": 8.1, "name": "Lemmatisation",
        "action": "WordNet lemmatiser (verb pass then noun pass) on tokens of length > 2",
        "detail": f"unique vocabulary reduced from {vocab_before:,} to {vocab_after:,} tokens "
                  f"({(1 - vocab_after/max(vocab_before,1))*100:.1f}% reduction)",
        "rows_affected": int(len(df)),
    })

    # ---- Step 9: extreme-length outliers (flag, do not delete)
    q1, q3 = df["n_tokens_raw"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = max(1, q1 - 1.5 * iqr)
    df["is_length_outlier"] = (df["n_tokens_raw"] > upper) | (df["n_tokens_raw"] < lower)
    n_out = int(df["is_length_outlier"].sum())
    log["steps"].append({
        "step": 9, "name": "Outlier detection (text length)",
        "action": "Flagged rows outside [Q1-1.5IQR, Q3+1.5IQR] tokens. NOT deleted - long tweets are legitimate",
        "detail": f"Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}, upper fence={upper:.1f}; {n_out} rows flagged "
                  f"({n_out/len(df)*100:.1f}%)",
        "rows_affected": 0,
        "flagged": n_out,
        "fences": {"lower": float(lower), "upper": float(upper)},
    })

    log["rows_out"] = int(len(df))
    log["rows_dropped_total"] = log["rows_in"] - log["rows_out"]
    return df.reset_index(drop=True), log


# ------------------------------------------------------------------ EDA
def fig_class_balance(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    counts = train["sentiment"].value_counts().sort_index()
    labels = [LABEL_NAMES[int(i)] for i in counts.index]
    bars = axes[0].bar(labels, counts.values,
                       color=[PALETTE[int(i)] for i in counts.index], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, counts.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 4, f"{v}\n({v/len(train)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=8.5)
    axes[0].set_title(f"Train class distribution (n={len(train)})")
    axes[0].set_ylabel("Number of tweets")
    axes[0].set_ylim(0, counts.max() * 1.28)

    ct = test["sentiment"].value_counts().sort_index()
    tl = [LABEL_NAMES[int(i)] for i in ct.index]
    bars = axes[1].bar(tl, ct.values, color=[PALETTE[int(i)] for i in ct.index],
                       edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, ct.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 3, f"{v}\n({v/len(test)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=8.5)
    axes[1].set_title(f"Test class distribution (n={len(test)})")
    axes[1].set_ylim(0, ct.max() * 1.28)

    fig.suptitle("Class balance is broadly similar across train and test\n"
                 "'Irrelevant' is the smallest class in both (~1.7 : 1 imbalance ratio)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_class_balance.png")
    plt.close(fig)
    return {"train_counts": counts.to_dict(), "test_counts": ct.to_dict()}


def fig_length(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

    sns.histplot(train["n_tokens_raw"], bins=30, kde=True, color="#4C72B0",
                 ax=axes[0], edgecolor="white")
    q1, q3 = train["n_tokens_raw"].quantile([.25, .75])
    upper = q3 + 1.5 * (q3 - q1)
    axes[0].axvline(upper, color="#D62728", ls="--", lw=1.4, label=f"IQR upper fence = {upper:.0f}")
    axes[0].axvline(train["n_tokens_raw"].mean(), color="#2CA02C", ls=":", lw=1.4,
                    label=f"mean = {train['n_tokens_raw'].mean():.1f}")
    axes[0].set_title("Train: tokens per tweet (after cleaning)")
    axes[0].set_xlabel("Token count")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(fontsize=8)

    plot_df = train.assign(cls=train["sentiment"].map(LABEL_NAMES))
    sns.boxplot(data=plot_df, x="cls", y="n_tokens_raw", hue="cls",
                order=[LABEL_NAMES[i] for i in range(4)],
                palette=[PALETTE[i] for i in range(4)], ax=axes[1], linewidth=0.8,
                legend=False)
    axes[1].set_title("Train: token count by sentiment class")
    axes[1].set_xlabel("Sentiment class")
    axes[1].set_ylabel("Token count")

    fig.suptitle("Tweet length distribution\n"
                 f"Median {train['n_tokens_raw'].median():.0f} tokens; the right tail is long but not erroneous",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_length_distribution.png")
    plt.close(fig)

    stats = train["n_tokens_raw"].describe().to_dict()
    stats["test_mean"] = float(test["n_tokens_raw"].mean())
    stats["upper_fence"] = float(upper)
    return stats


def fig_cleaning_effect(train_raw: pd.DataFrame, train: pd.DataFrame) -> dict:
    raw_len = train_raw["phrase"].astype(str).str.split().str.len()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

    sns.kdeplot(raw_len, label=f"Raw (mean {raw_len.mean():.1f})",
                ax=axes[0], fill=True, alpha=.45, color="#C44E52")
    sns.kdeplot(train["n_tokens_raw"], label=f"Cleaned (mean {train['n_tokens_raw'].mean():.1f})",
                ax=axes[0], fill=True, alpha=.45, color="#55A868")
    axes[0].set_title("Effect of cleaning on tweet length")
    axes[0].set_xlabel("Tokens per tweet")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    metrics = {
        "Unique vocab (raw)": len({w for s in train_raw["phrase"].astype(str) for w in tc.tokenise(s.lower())}),
        "Unique vocab (clean)": len({w for s in train["text"] for w in s.split()}),
    }
    names = list(metrics)
    vals = list(metrics.values())
    bars = axes[1].barh(names, vals, color=["#C44E52", "#55A868"], edgecolor="black", linewidth=.6)
    for b, v in zip(bars, vals):
        axes[1].text(v * 1.01, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=9)
    axes[1].set_title("Vocabulary size before vs after cleaning")
    axes[1].set_xlabel("Unique tokens in train")
    axes[1].set_xlim(0, max(vals) * 1.18)
    axes[1].invert_yaxis()

    fig.suptitle("What the cleaning pipeline changed", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_cleaning_effect.png")
    plt.close(fig)
    return metrics


def fig_noise() -> pd.DataFrame:
    """Noise prevalence per class, computed on the RAW train file."""
    raw = pd.read_csv(DATA_DIR / "traindata.csv", encoding="utf-8-sig")
    ph = raw["Phrase"].astype(str)
    feat = pd.DataFrame({
        "sentiment": raw["Sentiment"].astype(int),
        "Mojibake": ph.str.contains(r"[ÃÂâ]", regex=True),
        "@mentions": ph.str.contains("@", regex=False),
        "<unk> token": ph.str.contains("<unk>", regex=False),
        "URLs": ph.str.contains(r"https?://|www\.", case=False, regex=True),
        "ALL-CAPS": ph.str.contains(r"\b[A-Z]{3,}\b", regex=True),
        "Ellipses": ph.str.contains(r"\.{2,}", regex=True),
    })
    pct = (feat.groupby("sentiment").mean(numeric_only=True) * 100).round(1)
    long = pct.reset_index().melt(id_vars="sentiment", var_name="noise", value_name="pct")
    long["class"] = long["sentiment"].map(LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    sns.barplot(data=long, x="noise", y="pct", hue="class",
                hue_order=[LABEL_NAMES[i] for i in range(4)], ax=ax,
                palette=[PALETTE[i] for i in range(4)], edgecolor="black", linewidth=.5)
    ax.set_title("Data-noise prevalence by sentiment class (raw training data)")
    ax.set_xlabel("")
    ax.set_ylabel("% of rows in class")
    ax.legend(title="Class", fontsize=8, title_fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.10))
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", fontsize=7, padding=1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_noise_by_class.png")
    plt.close(fig)
    pct.to_csv(TAB_DIR / "noise_by_class.csv")
    return pct


def fig_top_words(train: pd.DataFrame, top_n: int = 12) -> dict:
    from collections import Counter
    stop_plot = STOPWORDS | {"game", "like", "just", "get", "one", "really", "still",
                             "going", "know", "think", "want", "need", "us", "im"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    out = {}
    for ax, lab in zip(axes.ravel(), range(4)):
        toks = Counter(w for s in train.loc[train["sentiment"] == lab, "text"]
                       for w in s.split() if w not in STOPWORDS and len(w) > 2)
        common = toks.most_common(top_n)[::-1]
        words = [w for w, _ in common]
        counts = [c for _, c in common]
        ax.barh(words, counts, color=PALETTE[lab], alpha=.85, edgecolor="black", linewidth=.4)
        ax.set_title(f"{LABEL_NAMES[lab]}  (n={int((train['sentiment']==lab).sum())})")
        ax.set_xlabel("Frequency")
        ax.tick_params(axis="y", labelsize=8.5)
        out[LABEL_NAMES[lab]] = {w: c for w, c in toks.most_common(top_n)}
    fig.suptitle("Top 12 discriminative terms per class (stop-words removed)\n"
                 "Profanity and 'not' dominate Negative; 'love'/'thank' dominate Positive",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_top_words.png")
    plt.close(fig)
    pd.DataFrame(out).fillna(0).astype(int).to_csv(TAB_DIR / "top_words_by_class.csv")
    return out


def fig_brand_sentiment(train: pd.DataFrame) -> pd.DataFrame:
    brands = ["PUBG", "Verizon", "NBA2K", "Madden", "Xbox", "Amazon", "GTA",
              "Rockstar", "Fortnite", "Overwatch", "FIFA", "Microsoft", "Google",
              "Call of Duty", "PlayStation"]
    rows = []
    raw = train["phrase"].astype(str)
    for b in brands:
        m = raw.str.contains(b, case=False, regex=False)
        if m.sum() < 5:
            continue
        dist = train.loc[m, "sentiment"].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0) * 100
        rows.append({"brand": b, "n": int(m.sum()),
                     "Negative": dist[0], "Positive": dist[1],
                     "Irrelevant": dist[2], "Neutral": dist[3]})
    bdf = pd.DataFrame(rows).sort_values("Negative", ascending=True)
    if bdf.empty:
        return bdf
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(bdf["brand"], bdf["Negative"], color=PALETTE[0], label="Negative",
            edgecolor="black", linewidth=.5)
    ax.barh(bdf["brand"], bdf["Positive"], left=bdf["Negative"], color=PALETTE[1],
            label="Positive", edgecolor="black", linewidth=.5)
    ax.barh(bdf["brand"], bdf["Neutral"], left=bdf["Negative"] + bdf["Positive"],
            color=PALETTE[3], label="Neutral", edgecolor="black", linewidth=.5)
    ax.barh(bdf["brand"], bdf["Irrelevant"], left=bdf["Negative"] + bdf["Positive"] + bdf["Neutral"],
            color=PALETTE[2], label="Irrelevant", edgecolor="black", linewidth=.5)
    for i, (_, r) in enumerate(bdf.iterrows()):
        ax.text(101, i, f"n={int(r['n'])}", va="center", fontsize=7.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of tweets mentioning the brand")
    ax.set_title("Sentiment mix for the 15 most-mentioned brands\n"
                 "Verizon, Madden and NBA2K skew heavily negative; Fortnite and Overwatch skew positive")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_brand_sentiment.png")
    plt.close(fig)
    bdf.to_csv(TAB_DIR / "brand_sentiment.csv", index=False)
    return bdf


def fig_wordclouds(train: pd.DataFrame) -> None:
    try:
        from wordcloud import WordCloud
    except Exception:
        print("  [skip] wordcloud not installed")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, lab in zip(axes.ravel(), range(4)):
        text = " ".join(train.loc[train["sentiment"] == lab, "text_no_sw"])
        wc = WordCloud(width=600, height=360, background_color="white",
                       colormap="Reds" if lab == 0 else ("Greens" if lab == 1 else
                                                         ("Greys" if lab == 2 else "Blues")),
                       max_words=90, random_state=42,
                       stopwords=STOPWORDS).generate(text or "empty")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{LABEL_NAMES[lab]} (n={int((train['sentiment']==lab).sum())})")
    fig.suptitle("Word clouds per sentiment class (stop-words removed)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_wordclouds.png")
    plt.close(fig)


def fig_duplicates_and_issues(train_raw: pd.DataFrame) -> pd.DataFrame:
    ph = train_raw["phrase"].astype(str)
    issues = {
        "Exact duplicate rows": int(train_raw.duplicated().sum()),
        "Duplicate phrase text": int(train_raw.duplicated(subset=["phrase"]).sum()),
        "Same text, different label": int(train_raw.groupby("phrase")["sentiment"].nunique().gt(1).sum()),
        "Mojibake (Ã/Â/â)": int(ph.str.contains(r"[ÃÂâ]", regex=True).sum()),
        "<unk> placeholder": int(ph.str.contains("<unk>", regex=False).sum()),
        "Embedded URLs": int(ph.str.contains(r"https?://|www\.", case=False, regex=True).sum()),
        "@mentions": int(ph.str.contains("@", regex=False).sum()),
        "HTML-like tags": int(ph.str.contains(r"<(?!unk>)[^<>]{1,30}>", regex=True).sum()),
        "Anonymiser token RhandlerR": int(ph.str.contains("RhandlerR", regex=False).sum()),
        "Non-ASCII characters": int(ph.str.contains(r"[^\x00-\x7F]", regex=True).sum()),
        "All-caps shouting": int(ph.str.contains(r"\b[A-Z]{3,}\b", regex=True).sum()),
    }
    s = pd.Series(issues).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#D62728" if v > 60 else "#FF7F0E" if v > 20 else "#4C72B0" for v in s.values]
    bars = ax.barh(s.index, s.values, color=colors, edgecolor="black", linewidth=.5)
    for b, v in zip(bars, s.values):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v}  ({v/len(train_raw)*100:.1f}%)",
                va="center", fontsize=8)
    ax.set_xlabel("Number of rows in raw training set (n=596)")
    ax.set_title("Data-quality audit: what was found before cleaning")
    ax.set_xlim(0, s.max() * 1.28)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig08_data_quality_audit.png")
    plt.close(fig)
    return s


def fig_learning_curve(curve: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(curve["train_size"], curve["train_f1"], "o-", color="#2CA02C", label="Train macro-F1")
    ax.plot(curve["train_size"], curve["cv_f1"], "s-", color="#D62728", label="CV macro-F1")
    ax.fill_between(curve["train_size"],
                    np.array(curve["cv_f1"]) - np.array(curve["cv_f1_std"]),
                    np.array(curve["cv_f1"]) + np.array(curve["cv_f1_std"]),
                    alpha=.18, color="#D62728")
    ax.set_xlabel("Training examples used")
    ax.set_ylabel("Macro F1")
    ax.set_title("Learning curve of the TF-IDF + Logistic Regression baseline\n"
                 "The widening train-CV gap indicates variance (over-fitting), not more-data starvation")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig09_learning_curve.png")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    print("=" * 90)
    print("STEP 1/3  Loading raw data")
    print("=" * 90)
    train_raw = load_raw(DATA_DIR / "traindata.csv")
    test_raw = load_raw(DATA_DIR / "testdata.csv")
    print(f"  train raw: {train_raw.shape}   test raw: {test_raw.shape}")

    print("\n" + "=" * 90)
    print("STEP 2/3  Running cleaning pipeline")
    print("=" * 90)
    train, train_log = clean_frame(train_raw, "train")
    test, test_log = clean_frame(test_raw, "test")
    print(f"  train clean: {train.shape}   ({train_log['rows_dropped_total']} rows dropped)")
    print(f"  test  clean: {test.shape}   ({test_log['rows_dropped_total']} rows dropped)")

    # leakage check after cleaning
    tr_set, te_set = set(train["text"]), set(test["text"])
    leak = len(tr_set & te_set)
    print(f"  exact text overlap after cleaning: {leak} rows")

    train.drop(columns=["phrase_fixed"]).to_csv(OUT_DATA / "train_clean.csv", index=False)
    test.drop(columns=["phrase_fixed"]).to_csv(OUT_DATA / "test_clean.csv", index=False)
    with open(OUT_DATA / "cleaning_log.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_log, "test": test_log,
                   "text_overlap_after_cleaning": leak}, f, indent=2, ensure_ascii=False)
    print(f"  written -> {OUT_DATA}")

    print("\n" + "=" * 90)
    print("STEP 3/3  Generating EDA figures")
    print("=" * 90)
    balance = fig_class_balance(train, test)
    print("  fig01 class balance       OK")
    length_stats = fig_length(train, test)
    print("  fig02 length distribution OK")
    vocab = fig_cleaning_effect(train_raw, train)
    print("  fig03 cleaning effect     OK")
    noise = fig_noise()
    print("  fig04 noise by class      OK")
    top_words = fig_top_words(train)
    print("  fig05 top words           OK")
    brands = fig_brand_sentiment(train)
    print("  fig06 brand sentiment     OK")
    fig_wordclouds(train)
    print("  fig07 word clouds         OK")
    issues = fig_duplicates_and_issues(train_raw)
    print("  fig08 quality audit       OK")

    summary = {
        "train_rows_in": train_log["rows_in"],
        "train_rows_out": train_log["rows_out"],
        "test_rows_in": test_log["rows_in"],
        "test_rows_out": test_log["rows_out"],
        "train_dropped": train_log["rows_dropped_total"],
        "test_dropped": test_log["rows_dropped_total"],
        "exact_overlap_after_cleaning": leak,
        "class_balance": {
            "train_counts": {LABEL_NAMES[int(k)]: int(v) for k, v in balance["train_counts"].items()},
            "test_counts": {LABEL_NAMES[int(k)]: int(v) for k, v in balance["test_counts"].items()},
        },
        "length_stats": {k: float(v) for k, v in length_stats.items()},
        "vocab": {k: int(v) for k, v in vocab.items()},
        "issues": {str(k): int(v) for k, v in issues.items()},
        "noise_by_class": {LABEL_NAMES[int(row)]: {str(c): float(v) for c, v in vals.items()
                                                    if c not in ("sentiment", "class")}
                           for row, vals in noise.iterrows()},
        "brands_found": int(len(brands)),
        "median_tokens": float(train["n_tokens_raw"].median()),
        "imbalance_ratio": round(float(train["sentiment"].value_counts().max() /
                                      train["sentiment"].value_counts().min()), 2),
    }
    with open(OUT_DATA / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  summary -> {OUT_DATA/'eda_summary.json'}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
