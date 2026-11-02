"""
Build the submission Jupyter notebook (.ipynb).

The brief requires "all source code, data preprocessing steps, and model
implementations ... in a single, well-annotated Jupyter Notebook". This script
assembles that notebook programmatically so the JSON is always valid, then the
notebook is executed by build_notebook.py to embed real outputs.

Run: python src/make_notebook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DSAI4205_Interim_Group8_Twitter_Sentiment.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


CELLS: list[dict] = []

# ---------------------------------------------------------------- title
CELLS.append(md("""# DSAI4205 Big Data Analytics - Group Project
## Dataset 5: Twitter Entity-Level Sentiment Analysis
### Interim Technical Notebook - EDA, Data Cleaning and TF-IDF + Logistic Regression Baseline

**Group 8 - ChillGuy**

| Name | Student ID |
|---|---|
| Chao Ka Kin | 23021004D |
| Leung Wai Tsun | 24097714D |
| LIANG Zijian | 24082468D |
| Lam Kin Ho | 24084386D |
| Yeung Tsz Chun Herman | 23019365D |

---

### How to read this notebook

This notebook is the **technical deliverable** for the interim submission. It is organised in the
same order as the written report:

| Section | Content | Report section |
|---|---|---|
| 0 | Setup and reproducibility | 9 |
| 1 | Load data and inspect the raw files | 2 |
| 2 | Exploratory Data Analysis | 3 |
| 3 | Data cleaning pipeline | 4 |
| 4 | Baseline model: TF-IDF + Logistic Regression | 5 |
| 5 | Error analysis | 6 |
| 6 | Findings and open questions | 7 |

Every figure and number in the written report is produced by the cells below.
The notebook is deterministic: the random seed is fixed at 42 throughout.
"""))

CELLS.append(md("""---
## 0. Setup and reproducibility

We pin the random seed and define the project paths once, so that re-running the notebook
top-to-bottom reproduces every result exactly.
"""))

CELLS.append(code('''import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 110)

# ---------------------------------------------------------------- paths
ROOT = Path.cwd()
DATA_DIR = ROOT / "Data 5 - Twitter" / "Data 5 - Twitter"
TRAIN_CSV = DATA_DIR / "traindata.csv"
TEST_CSV = DATA_DIR / "testdata.csv"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"

# import our own cleaning module so the notebook and the pipeline cannot diverge
sys.path.insert(0, str(ROOT / "src"))
import text_clean as tc

print("Python      :", sys.version.split()[0])
print("pandas      :", pd.__version__)
print("numpy       :", np.__version__)
print("matplotlib  :", matplotlib.__version__)
print("seaborn     :", sns.__version__)
'''))

CELLS.append(code('''import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

print("scikit-learn:", sklearn.__version__)

plt.rcParams.update({
    "figure.dpi": 110,
    "figure.figsize": (10, 4.2),
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 9,
})
sns.set_theme(style="whitegrid", context="notebook")

LABEL_NAMES = {0: "Negative", 1: "Positive", 2: "Irrelevant", 3: "Neutral"}
PALETTE = {0: "#D62728", 1: "#2CA02C", 2: "#7F7F7F", 3: "#1F77B4"}
'''))

# ---------------------------------------------------------------- 1. load
CELLS.append(md("""---
## 1. Load the data and understand what we were given

### 1.1 Business framing

The brief asks for a **decision support system that classifies text**. For Twitter sentiment the
practical use is brand and product monitoring: route complaints, spot emerging product failures,
and separate genuine opinion from unrelated chatter.

### 1.2 Type of data and granularity

- **Type:** unstructured text (`Phrase`) plus a categorical target (`Sentiment`).
- **Granularity:** one row = one tweet. Tweets are truncated at roughly 350 characters.
- **No identifiers:** there is no user, timestamp, or entity column.

That last point matters more than it first appears - we return to it in the findings.
"""))

CELLS.append(code('''train_raw = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
test_raw = pd.read_csv(TEST_CSV, encoding="utf-8-sig")

print(f"traindata.csv : {train_raw.shape[0]:>4} rows x {train_raw.shape[1]} columns")
print(f"testdata.csv  : {test_raw.shape[0]:>4} rows x {test_raw.shape[1]} columns")
print()
print("Columns in traindata.csv:", list(train_raw.columns))
print("Columns in testdata.csv :", list(test_raw.columns))
print()
print("dtypes:")
print(train_raw.dtypes.to_string())
print()
train_raw.head(8)
'''))

CELLS.append(md("""### 1.3 The schema does not match the documentation

The supplied `Twitter Sentiment Analysis Dataset.docx` states:

> "This is an **entity-level** sentiment analysis dataset of twitter. Given a **message and an
> entity**, the task is to judge the sentiment of the message **about the entity**. There are four
> classes: Negative, Positive, Irrelevant and Neutral."

But the delivered CSVs contain **only two columns** - `Phrase` and `Sentiment`. There is **no entity
column**.

This is not a cosmetic detail. "Irrelevant" is defined as *"the message is not about the entity"*.
With no entity recorded, that label cannot be applied consistently - not by a human annotator and
certainly not by a model. We document the deviation rather than silently inventing an entity.

Let us verify this programmatically so the finding is evidenced, not asserted.
"""))

CELLS.append(code('''expected_beyond_phrase_and_label = [
    c for c in list(train_raw.columns) + list(test_raw.columns)
    if c.lower() in {"entity", "topic", "brand", "company", "subject"}
]

print("Entity-like columns found:", expected_beyond_phrase_and_label or "NONE")
print()
print("Documentation promises : message + ENTITY -> 4-class verdict")
print("Delivered columns      :", list(train_raw.columns))
print()
print("=> The 'Irrelevant' label is defined relative to information that is absent from the file.")
'''))

CELLS.append(md("""### 1.4 Missing values, duplicates and label domain

Before any modelling we need to know what is broken. We check missing values, blank strings,
exact duplicates, contradictory labels, and whether the label domain matches the documented 0-3.
"""))

CELLS.append(code('''def audit(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Compact data-quality audit of one raw file."""
    ph = df["Phrase"].astype(str)
    rows = {
        "rows": len(df),
        "missing Phrase": int(df["Phrase"].isna().sum()),
        "missing Sentiment": int(df["Sentiment"].isna().sum()),
        "blank after strip": int((ph.str.strip() == "").sum()),
        "exact duplicate rows": int(df.duplicated().sum()),
        "duplicate Phrase text": int(df.duplicated(subset=["Phrase"]).sum()),
        "Phrase with >1 label": int(df.groupby("Phrase")["Sentiment"].nunique().gt(1).sum()),
        "invalid labels (outside 0-3)": int((~df["Sentiment"].isin([0, 1, 2, 3])).sum()),
        "labels observed": sorted(df["Sentiment"].dropna().unique().tolist()),
        "Sentiment dtype": str(df["Sentiment"].dtype),
    }
    return pd.DataFrame({"check": list(rows), name: list(rows.values())}).set_index("check")


audit_table = audit(train_raw, "train").join(audit(test_raw, "test"), how="left")
display(audit_table)
'''))

CELLS.append(md("""**Interpretation.** There are **no missing values and no invalid labels**, which is good. But there
*are* duplicates, and one phrase carries two different labels. We also notice the label dtype is
inconsistent (`float64` in train because of the `0.0` formatting, `int64` in test) - trivial, but it
will silently break any naive concatenation of the two files, so it must be standardised.

### 1.5 Text-level quality defects

Beyond the tabular checks, text carries its own class of defects. We scan for the artefacts that
actually damage tokenisation.
"""))

CELLS.append(code('''ph = train_raw["Phrase"].astype(str)

defect_checks = {
    "Mojibake (double-encoded UTF-8)": r"[ÃÂâ]",
    "Non-ASCII characters": r"[^\\x00-\\x7F]",
    "<unk> placeholder": None,
    "Embedded URLs": r"https?://|www\\.",
    "@mentions": None,
    "HTML-like tags": r"<(?!unk>)[^<>]{1,30}>",
    "Anonymiser token 'RhandlerR'": None,
    "ALL-CAPS shouting (3+)": r"\\b[A-Z]{3,}\\b",
    "Ellipses (2+ dots)": r"\\.{2,}",
}

results = []
for label, pattern in defect_checks.items():
    if pattern is None:
        needle = {"<unk> placeholder": "<unk>", "@mentions": "@",
                  "Anonymiser token 'RhandlerR'": "RhandlerR"}[label]
        n = int(ph.str.contains(needle, regex=False).sum())
    else:
        n = int(ph.str.contains(pattern, regex=True).sum())
    results.append({"defect": label, "rows": n, "pct": round(n / len(ph) * 100, 1)})

defects = pd.DataFrame(results).sort_values("rows", ascending=False).reset_index(drop=True)
display(defects)

print("Example of mojibake (the bytes for a curly apostrophe were decoded as cp1252):")
for s in ph[ph.str.contains(r"[ÃÂâ]", regex=True)].head(2):
    print("   ", s[:105])
'''))

CELLS.append(md("""**What mojibake actually is.** The file was written as UTF-8 but read as cp1252, so a single curly
apostrophe `'` (3 bytes) became three visible characters `â€™`. It is *recoverable* corruption, and
left unrepaired it splits every affected word out of the vocabulary: `I'm` and `Iâm` become two
unrelated tokens. We repair it in Section 3.

**What `<unk>` is.** A leftover placeholder from whatever tokeniser produced the file. Sometimes it
is glued to a real word (`rework<unk>`), so naive removal would create a non-word (`reworkunk`).
"""))

# ---------------------------------------------------------------- 2. EDA
CELLS.append(md("""---
## 2. Exploratory Data Analysis

### 2.1 Class distribution and imbalance

The first question in any classification project is whether the classes are balanced. Severe
imbalance would mean accuracy is a misleading metric and would require resampling.
"""))

CELLS.append(code('''fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

for ax, df, name in [(axes[0], train_raw, "Train"), (axes[1], test_raw, "Test")]:
    counts = df["Sentiment"].astype(int).value_counts().sort_index()
    labels = [LABEL_NAMES[i] for i in counts.index]
    bars = ax.bar(labels, counts.values,
                  color=[PALETTE[i] for i in counts.index],
                  edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + counts.max() * 0.02,
                f"{v}\\n({v/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{name} class distribution (n={len(df)})")
    ax.set_ylabel("Number of tweets")
    ax.set_ylim(0, counts.max() * 1.30)

plt.suptitle("Class balance is similar across train and test", fontweight="bold")
plt.tight_layout()
plt.show()

tr_counts = train_raw["Sentiment"].astype(int).value_counts().sort_index()
imbalance = tr_counts.max() / tr_counts.min()
print(f"RAW file  -> largest/smallest class ratio : {imbalance:.2f} : 1")
print(f"RAW file  -> majority-class accuracy       : {tr_counts.max()/len(train_raw)*100:.1f}% "
      f"(always predicting '{LABEL_NAMES[int(tr_counts.idxmax())]}')")
'''))

CELLS.append(md("""**Finding: imbalance is mild (1.65 : 1), not severe.** This is well below the 10:1 ratio where
resampling becomes necessary, so we deliberately do **not** apply SMOTE. The train and test
distributions also agree closely, which means a single hold-out evaluation is trustworthy.

This matters for the headline metric: with only mild imbalance, **accuracy is a reasonable
companion to macro F1** - unlike the situation where 95% of rows share one class.
"""))

CELLS.append(md("""### 2.2 Tweet length distribution and outliers

Length is the only continuous quantity we have. We look at its distribution to decide two things:
whether to cap tweet length, and whether "outliers" here are errors or genuine content.
"""))

CELLS.append(code('''raw_words = train_raw["Phrase"].astype(str).str.split().str.len()
q1, q3 = raw_words.quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr

fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

sns.histplot(raw_words, bins=28, kde=True, color="#4C72B0", ax=axes[0], edgecolor="white")
axes[0].axvline(upper_fence, color="#D62728", ls="--", lw=1.4,
                label=f"1.5xIQR fence = {upper_fence:.0f} words")
axes[0].axvline(raw_words.mean(), color="#2CA02C", ls=":", lw=1.4,
                label=f"mean = {raw_words.mean():.1f}")
axes[0].set_title("Words per tweet (raw)")
axes[0].set_xlabel("Word count")
axes[0].set_ylabel("Frequency")
axes[0].legend(fontsize=8)

plot_df = train_raw.assign(cls=train_raw["Sentiment"].astype(int).map(LABEL_NAMES))
sns.boxplot(data=plot_df, x="cls", y=plot_df["Phrase"].astype(str).str.split().str.len(),
            hue="cls", order=[LABEL_NAMES[i] for i in range(4)],
            palette=[PALETTE[i] for i in range(4)], ax=axes[1], linewidth=0.8, legend=False)
axes[1].set_title("Words per tweet by class")
axes[1].set_xlabel("Sentiment class")
axes[1].set_ylabel("Word count")

plt.suptitle("Tweet length is right-skewed but the tail is genuine content", fontweight="bold")
plt.tight_layout()
plt.show()

n_outliers = int((raw_words > upper_fence).sum())
print(raw_words.describe().round(2).to_string())
print(f"\\nRows beyond the 1.5xIQR upper fence: {n_outliers} "
      f"({n_outliers/len(raw_words)*100:.1f}%) - FLAGGED, not removed")
'''))

CELLS.append(md("""**Decision: flag length outliers, do not delete them.**

This is a deliberate departure from the textbook 1.5xIQR rule, and the reasoning is domain-specific:

- In *tabular* data an outlier is often a measurement error, so removing it is defensible.
- In *text* data a long tweet is almost always a real, detailed complaint - precisely the subclass
  a brand-monitoring system must **not** lose. Deleting the long tail would systematically blind the
  model to customers who explain their problem at length.

We therefore keep the rows and expose an `is_length_outlier` flag so any downstream model can cap
or weight them if it wishes. The medians across classes barely differ, so length alone carries
little class signal.
"""))

CELLS.append(md("""### 2.3 Correlation between variables

With one text column and one categorical target, a conventional correlation matrix has nothing to
correlate. The correct analogue for unstructured data is to measure how **engineered text signals**
associate with the label.
"""))

CELLS.append(code('''def brand_regex() -> str:
    brands = ["PUBG", "Amazon", "Verizon", "CallofDuty", "Call of Duty", "Xbox", "PlayStation",
              "Microsoft", "Fortnite", "Overwatch", "NBA2K", "FIFA", "Madden", "Google",
              "Apple", "Facebook", "Twitter", "Rockstar", "GTA", "Blizzard", "Nintendo"]
    return "|".join(re.escape(b) for b in brands)

phrase = train_raw["Phrase"].astype(str)
signals = pd.DataFrame({
    "sentiment": train_raw["Sentiment"].astype(int),
    "has @mention": phrase.str.contains("@", regex=False),
    "has URL": phrase.str.contains(r"https?://|www\\.", case=False, regex=True),
    "has <unk>": phrase.str.contains("<unk>", regex=False),
    "has mojibake": phrase.str.contains(r"[ÃÂâ]", regex=True),
    "has ALL-CAPS": phrase.str.contains(r"\\b[A-Z]{3,}\\b", regex=True),
    "has ellipsis": phrase.str.contains(r"\\.{2,}", regex=True),
    "mentions a brand": phrase.str.contains(brand_regex(), case=False, regex=True),
    "first person": phrase.str.contains(r"\\b(?:i|my|me|im)\\b", case=False, regex=True),
    "profanity": phrase.str.contains(r"\\b(?:fuck|shit|damn|hell|ass)\\w*", case=False, regex=True),
})

signal_pct = (signals.groupby("sentiment").mean(numeric_only=True) * 100).round(1)
signal_pct.index = [LABEL_NAMES[i] for i in signal_pct.index]
display(signal_pct)

# correlation of each binary signal with "is negative"
signals_neg = signals.copy()
signals_neg["is_negative"] = (signals_neg["sentiment"] == 0).astype(int)
corr = (signals_neg.drop(columns="sentiment")
        .corr()["is_negative"].drop("is_negative").sort_values())
print("\\nCorrelation with 'is negative':")
print(corr.round(3).to_string())
'''))

CELLS.append(md("""**Reading the signal table.** Two signals are genuinely informative:

- **@mentions** appear in 38% of Negative rows but only 18% of Neutral rows - negative tweets are
  usually addressed *at* a company.
- **URLs** are almost absent from Negative rows (0.6%) and concentrated in Neutral and Irrelevant
  rows (8.8% and 4.7%) - those classes have a news-and-links character.

**Mojibake is flat across all four classes** (17 / 10 / 17 / 19 %). That flatness is the proof that
it is random technical corruption rather than semantic signal - which is exactly why removing it
cannot bias any class. This is a useful habit: before deleting a "noise" feature, check whether its
distribution is class-dependent. If it were, deleting it could remove real signal.

### 2.4 Which words actually separate the classes?

We profile the most frequent discriminative terms per class. This tells us how separable the
problem is *before* we fit any model.
"""))

CELLS.append(code('''from collections import Counter

EXTRA_STOP = {"http", "https", "www", "com", "amp", "rt", "rhandlerr", "pic",
              "twitter", "t", "co", "unk", "via", "lol", "lmao"}
STOPWORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOP

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, lab in zip(axes.ravel(), range(4)):
    tokens = Counter(
        w for s in phrase[train_raw["Sentiment"].astype(int) == lab]
        for w in re.findall(r"[a-z']{3,}", s.lower())
        if w not in STOPWORDS
    )
    common = tokens.most_common(12)[::-1]
    ax.barh([w for w, _ in common], [c for _, c in common],
            color=PALETTE[lab], alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_title(f"{LABEL_NAMES[lab]} (n={int((train_raw['Sentiment'].astype(int)==lab).sum())})")
    ax.set_xlabel("Frequency")
    ax.tick_params(axis="y", labelsize=8)

plt.suptitle("Top discriminative terms per class (stop-words removed)", fontweight="bold")
plt.tight_layout()
plt.show()
'''))

CELLS.append(md("""**Interpretation.** Negative and Positive separate cleanly on *affect* vocabulary (profanity /
failure words vs. `love` / `favorite` / `fun`). Neutral and Irrelevant, however, both look like
*news*: they share named entities, links and reporting verbs.

That overlap is the structural reason the baseline will struggle on the Irrelevant class, and we
should expect it before seeing the results rather than treating it as a surprise later.
"""))

CELLS.append(md("""### 2.5 Brand-level view: the decision-support angle

A classifier is only useful when its output is aggregated to something a manager cares about. We
therefore check whether sentiment separates by brand even though the model never sees the brand
column explicitly.
"""))

CELLS.append(code('''brands = ["PUBG", "Verizon", "NBA2K", "Madden", "Xbox", "Amazon", "GTA",
          "Rockstar", "Fortnite", "Overwatch", "FIFA", "Microsoft", "Google",
          "Call of Duty", "PlayStation"]

rows = []
for b in brands:
    mask = phrase.str.contains(b, case=False, regex=False)
    if mask.sum() < 5:
        continue
    dist = (train_raw.loc[mask, "Sentiment"].astype(int)
            .value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0) * 100)
    rows.append({"brand": b, "n": int(mask.sum()),
                 "Negative": dist[0], "Positive": dist[1],
                 "Irrelevant": dist[2], "Neutral": dist[3]})

brand_df = pd.DataFrame(rows).sort_values("Negative", ascending=True)
display(brand_df.round(1))

fig, ax = plt.subplots(figsize=(10, 5))
left = np.zeros(len(brand_df))
for cls, colour in [(0, PALETTE[0]), (1, PALETTE[1]), (3, PALETTE[3]), (2, PALETTE[2])]:
    ax.barh(brand_df["brand"], brand_df[LABEL_NAMES[cls]], left=left,
            color=colour, label=LABEL_NAMES[cls], edgecolor="black", linewidth=0.5)
    left += brand_df[LABEL_NAMES[cls]].values
for i, (_, r) in enumerate(brand_df.iterrows()):
    ax.text(101, i, f"n={int(r['n'])}", va="center", fontsize=7.5)
ax.set_xlim(0, 112)
ax.set_xlabel("% of tweets mentioning the brand")
ax.set_title("Sentiment mix by brand", fontweight="bold")
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.tight_layout()
plt.show()
'''))

CELLS.append(md("""**This is the business insight.** Verizon (64% negative), Madden (69%) and NBA2K (70%) generate
the most hostility, while Fortnite (18%) and Overwatch (21%) sit at the positive end. A brand
manager reading this chart knows immediately where to send the support team.

Note the sample sizes - most brands have fewer than 25 tweets, so a single tweet moves a percentage
by 4-5 points. The ranking is indicative, not statistically established. We report it as a
*direction* and would insist on more data before acting on a single brand.
"""))

# ---------------------------------------------------------------- 3. cleaning
CELLS.append(md("""---
## 3. Data Cleaning Pipeline

Each step below records **what** it changed and **how many rows** it touched. The same functions
are applied to train and test, so the two files can never drift apart.

### 3.1 Step-by-step cleaning

The chain is implemented in `src/text_clean.py` (imported at the top of this notebook) so that this
notebook and the production pipeline share exactly one implementation.
"""))

CELLS.append(code('''STOPWORDS = set(ENGLISH_STOP_WORDS) | {"http", "https", "www", "com", "amp", "rt",
                                          "rhandlerr", "pic", "twitter", "t", "co",
                                          "unk", "via", "lol", "lmao"}

CONTEXT = {"0": "Negative", "1": "Positive", "2": "Irrelevant", "3": "Neutral"}

def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the full documented cleaning pipeline to one raw dataframe."""
    log: list[dict] = []
    # standardise the schema first so train (float labels) and test (int labels)
    # can be handled by identical code below
    df = df.rename(columns={"Phrase": "phrase", "Sentiment": "sentiment"}).copy()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").astype("Int64")
    rows_in = len(df)
    log.append({"step": 1, "name": "Schema standardisation",
                "detail": "Phrase->phrase, Sentiment->sentiment, label cast to Int64",
                "affected": len(df)})

    # missing values / blanks
    before = len(df)
    df = df[df["phrase"].notna() & df["sentiment"].notna()]
    df = df[df["phrase"].astype(str).str.strip() != ""]
    log.append({"step": 2, "name": "Missing / blank removal",
                "detail": "dropped rows with no text or no label", "affected": before - len(df)})

    # exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["phrase", "sentiment"], keep="first")
    log.append({"step": 3, "name": "Exact duplicate removal",
                "detail": "kept first occurrence of each (text, label) pair",
                "affected": before - len(df)})

    # conflicting labels -> majority vote
    before = len(df)
    nun = df.groupby("phrase")["sentiment"].transform("nunique")
    conflict_mask = nun > 1
    n_conflict_text = df.loc[conflict_mask, "phrase"].nunique()
    if conflict_mask.any():
        majority = (df[conflict_mask].groupby("phrase")["sentiment"]
                    .agg(lambda s: s.value_counts().idxmax()))
        drop_idx = []
        for p, lab in majority.items():
            members = df[df["phrase"] == p]
            same = members[members["sentiment"] == lab]
            drop_idx += same.index[1:].tolist()
            drop_idx += members[members["sentiment"] != lab].index.tolist()
        df = df.drop(index=drop_idx)
    log.append({"step": 4, "name": "Conflicting-label resolution",
                "detail": f"{n_conflict_text} phrase(s) had contradictory labels; kept the majority",
                "affected": before - len(df)})

    # encoding repair
    moji_before = int(df["phrase"].astype(str).str.contains(r"[ÃÂâ]", regex=True).sum())
    df["phrase_fixed"] = df["phrase"].astype(str).map(tc.fix_encoding)
    moji_after = int(df["phrase_fixed"].str.contains(r"[ÃÂâ]", regex=True).sum())
    log.append({"step": 5, "name": "Encoding repair (mojibake)",
                "detail": f"ftfy + NFKC; rows containing 'Ã/Â/â': {moji_before} -> {moji_after}",
                "affected": moji_before - moji_after})

    # normalisation + tokenisation + stop-words + lemma
    df["text"] = df["phrase_fixed"].map(tc.basic_clean)
    df["text_no_sw"] = df["text"].map(
        lambda s: " ".join(tc.remove_stopwords(tc.tokenise(s), STOPWORDS)))
    lem = tc._get_lemmatiser()
    df["text_lemma"] = df["text_no_sw"].map(
        lambda s: " ".join(tc.lemmatise(s.split(), lem)))
    df["n_tokens"] = df["text"].map(tc.token_count)

    vocab_before = len({w for s in df["text_no_sw"] for w in s.split()})
    vocab_after = len({w for s in df["text_lemma"] for w in s.split()})
    log.append({"step": 6, "name": "Normalisation + tokenisation",
                "detail": "lowercase, contractions expanded, URLs/@mentions/tags/<unk> stripped, "
                          "elongation collapsed, stop-words removed (negations preserved)",
                "affected": len(df)})
    log.append({"step": 7, "name": "Lemmatisation",
                "detail": f"WordNet (verb then noun); vocabulary {vocab_before:,} -> {vocab_after:,}",
                "affected": len(df)})

    # length outliers flagged, not dropped
    q1, q3 = df["n_tokens"].quantile([0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    df["is_length_outlier"] = df["n_tokens"] > fence
    log.append({"step": 8, "name": "Outlier flagging",
                "detail": f"1.5xIQR upper fence = {fence:.1f} tokens; flagged "
                          f"{int(df['is_length_outlier'].sum())} rows (not removed)",
                "affected": 0})

    df = df.reset_index(drop=True)
    log.append({"step": 9, "name": "Summary",
                "detail": f"{rows_in} -> {len(df)} rows ({rows_in-len(df)} dropped)",
                "affected": rows_in - len(df)})
    return df, log


train, train_log = clean_frame(train_raw)
test, test_log = clean_frame(test_raw)
display(pd.DataFrame(train_log).set_index("step"))

cleaned_counts = train["sentiment"].astype(int).value_counts().sort_index()
print(f"CLEANED train -> ratio = {cleaned_counts.max()/cleaned_counts.min():.2f} : 1, "
      f"majority-class accuracy = {cleaned_counts.max()/len(train)*100:.1f}% "
      f"(raw file gave {train_raw['Sentiment'].astype(int).value_counts().max()/train_raw['Sentiment'].astype(int).value_counts().min():.2f} : 1 "
      f"and {train_raw['Sentiment'].astype(int).value_counts().max()/len(train_raw)*100:.1f}%)")
'''))

CELLS.append(code('''print("TEST file cleaning log")
print("-" * 78)
display(pd.DataFrame(test_log).set_index("step"))
'''))

CELLS.append(md("""### 3.2 Before / after evidence

A cleaning pipeline is only credible if we can show concrete transformations. These are real rows
from the dataset, chosen to exercise the hardest cases.
"""))

CELLS.append(code('''examples = train_raw.loc[
    train_raw["Phrase"].astype(str).str.contains(r"<unk>|[ÃÂâ]", regex=True)
].head(6)

print(f"{'RAW':<62} | CLEANED")
print("-" * 128)
for s in examples["Phrase"].astype(str):
    cleaned = tc.basic_clean(s)
    lem = " ".join(tc.lemmatise(
        tc.remove_stopwords(tc.tokenise(cleaned), STOPWORDS), tc._get_lemmatiser()))
    print(f"{s[:60]:<62} | {lem[:60]}")
'''))

CELLS.append(md("""**What the four behaviours in that output demonstrate:**

1. **Mojibake repaired** - `Iâm` becomes `i am`, so the token re-joins the rest of the corpus.
2. **`<unk>` removed cleanly** - `rework<unk>` becomes `rework`, not `reworkunk`. The regex uses a
   lookbehind/lookahead so an adjacent letter is never welded on.
3. **Negation preserved** - `can't` expands to `can not` (before punctuation removal), and `not`
   survives stop-word removal. A pipeline that strips `not` would turn "I can not make it work" into
   "I make it work" and invert the sentiment.
4. **Elongation collapsed** - `sooooo` becomes `soo`, keeping the word recognisable instead of
   creating a unique out-of-vocabulary token.

### 3.3 Why clean at all? (An honest answer)

We explicitly measured whether cleaning improves the score, rather than assuming it. Section 4.3
shows the result: **it barely moves macro F1**. Cleaning is justified by *correctness* (the data is
objectively corrupt), *test validity* (duplicates would leak across CV folds), and *downstream
compatibility* (BERT in Weeks 11-13 is highly sensitive to mojibake) - not by accuracy gains.

We also checked for leakage between the cleaned train and test sets.
"""))

CELLS.append(code('''# We check overlap on the actual model input (text_lemma), since that is what the
# classifier sees. More aggressive normalisation can merge distinct raw tweets into
# identical strings, so this is the conservative (stricter) check.
train_texts = set(train["text_lemma"].dropna())
test_texts = set(test["text_lemma"].dropna())
overlap = train_texts & test_texts

print(f"Train rows (cleaned)      : {len(train)}")
print(f"Test rows (cleaned)       : {len(test)}")
print(f"Exact overlap on model input: {len(overlap)} row(s) "
      f"({len(overlap)/len(test_texts)*100:.2f}% of test vocabulary strings)")
print()
for s in sorted(overlap)[:4]:
    print("   -", s[:95])
print()
print("Interpretation: overlap is a few rows out of 397, so the held-out score remains a")
print("fair out-of-sample estimate. We report it rather than hiding it.")
'''))

# ---------------------------------------------------------------- 4. model
CELLS.append(md("""---
## 4. Baseline Model: TF-IDF + Logistic Regression

### 4.1 Design

Phase 3i requires exactly one baseline: logistic regression on TF-IDF features. We keep it
deliberately plain so it is a fair reference point rather than a tuned competitor.

| Component | Setting | Why |
|---|---|---|
| Vectoriser | TF-IDF, word 1-2 grams | bigrams capture negation phrases such as "not good" |
| Term frequency | `sublinear_tf=True` | log-damped, so long tweets cannot dominate |
| Vocabulary | `min_df=1`, `max_df=0.95` | tiny corpus: rare terms are still informative |
| Classifier | multinomial logistic regression | required baseline; inherently interpretable |
| Regularisation | `C=1.0` (L2), lbfgs | untuned default, intentionally |
| Max iterations | 2000 | ensures convergence at this feature count |
"""))

CELLS.append(code('''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(balanced: bool = False) -> Pipeline:
    """TF-IDF -> multinomial logistic regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\\b[a-zA-Z][a-zA-Z']+\\b",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced" if balanced else None,
            random_state=RANDOM_STATE,
        )),
    ])


# quick look at what the vectoriser actually produces
probe = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                        token_pattern=r"(?u)\\b[a-zA-Z][a-zA-Z']+\\b").fit(train["text_lemma"])
print(f"TF-IDF vocabulary size: {len(probe.vocabulary_):,}")
print("Sample features:", list(probe.vocabulary_)[:18])
'''))

CELLS.append(md("""### 4.2 Evaluation protocol

With 592 training rows, a single train/validation split would produce a benchmark whose confidence
interval is wider than any improvement we could hope to measure. We therefore report:

1. **Stratified 5-fold cross-validation on the training file only** - the stable headline number,
   used for every model-selection decision.
2. **A single held-out evaluation on `testdata.csv`** - fitted on all 592 training rows. The test
   file is never used for model selection, so this is an unbiased out-of-sample estimate.

We lead with **macro F1** because accuracy alone hides the collapse of the minority class.
"""))

CELLS.append(code('''from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedKFold

y_train = train["sentiment"].astype(int)
y_test = test["sentiment"].astype(int)
X_train_lemma = train["text_lemma"].fillna("")
X_test_lemma = test["text_lemma"].fillna("")


def cv_evaluate(X, y, tag: str, balanced: bool = False, n_splits: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs, f1m, f1w = [], [], []
    for tr_i, va_i in skf.split(X, y):
        pipe = build_pipeline(balanced)
        pipe.fit(X.iloc[tr_i], y.iloc[tr_i])
        pred = pipe.predict(X.iloc[va_i])
        accs.append(accuracy_score(y.iloc[va_i], pred))
        f1m.append(f1_score(y.iloc[va_i], pred, average="macro"))
        f1w.append(f1_score(y.iloc[va_i], pred, average="weighted"))
    return {
        "variant": tag,
        "accuracy": np.mean(accs), "accuracy_sd": np.std(accs),
        "macro_f1": np.mean(f1m), "macro_f1_sd": np.std(f1m),
        "weighted_f1": np.mean(f1w),
    }
'''))

CELLS.append(md("""### 4.3 Does cleaning actually help? (measured, not assumed)

We run four variants of the same model with different inputs. This converts each cleaning decision
into a measured effect instead of a claim.
"""))

CELLS.append(code('''variants = [
    ("V1 raw text (no cleaning)",   train_raw["Phrase"].astype(str).iloc[:len(train)], False),
    ("V2 cleaned",                  train["text"].fillna(""),                        False),
    ("V2b cleaned + stop-words",    train["text_no_sw"].fillna(""),                  False),
    ("V2c + lemmatised",            X_train_lemma,                                   False),
    ("V3 + balanced class weight",  X_train_lemma,                                   True),
]

cv_rows = []
for tag, X, bal in variants:
    r = cv_evaluate(X, y_train, tag, balanced=bal)
    cv_rows.append(r)
    print(f"{tag:<32} acc={r['accuracy']*100:5.1f}%  macro-F1={r['macro_f1']*100:5.1f}% "
          f"(+/-{r['macro_f1_sd']*100:4.1f})  weighted-F1={r['weighted_f1']*100:5.1f}%")

cv_df = pd.DataFrame(cv_rows)
display(cv_df.round(4))
'''))

CELLS.append(code('''fig, ax = plt.subplots(figsize=(11, 4.2))
melted = cv_df.melt(id_vars="variant", value_vars=["accuracy", "macro_f1", "weighted_f1"],
                    var_name="metric", value_name="score")
melted["metric"] = melted["metric"].map({"accuracy": "Accuracy", "macro_f1": "Macro F1",
                                         "weighted_f1": "Weighted F1"})
sns.barplot(data=melted, x="variant", y="score", hue="metric", ax=ax,
            palette=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black", linewidth=0.5)
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=7, padding=1.5)
ax.set_ylim(0, 0.62)
ax.set_xlabel("")
ax.set_ylabel("5-fold CV score on train")
ax.set_title("Cleaning decisions expressed as measured effects", fontweight="bold")
ax.tick_params(axis="x", labelsize=8)
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
plt.tight_layout()
plt.show()
'''))

CELLS.append(md("""**Honest reading of this table.** Cleaning alone changes macro F1 by roughly zero within the
fold-to-fold noise band (+/- 5 pp). We report that rather than claiming a spurious improvement.

The one clear winner is **class-balanced weighting**: macro F1 rises from 39.4% to 46.0% while
accuracy stays flat. That is the single largest lever we found, and it directly previews the
Phase 3iv "handling imbalance" requirement.
"""))

CELLS.append(md("""### 4.4 Final baseline: held-out test performance

Now we fit the model on all 592 training rows and score it once on the 397-row test file.
"""))

CELLS.append(code('''final = build_pipeline(balanced=False).fit(X_train_lemma, y_train)
y_pred = final.predict(X_test_lemma)
y_proba = final.predict_proba(X_test_lemma)

test_acc = accuracy_score(y_test, y_pred)
test_f1m = f1_score(y_test, y_pred, average="macro")
test_f1w = f1_score(y_test, y_pred, average="weighted")

majority = int(y_train.value_counts().idxmax())
maj_acc = float((y_test == majority).mean())

print("HELD-OUT TEST RESULTS  (TF-IDF + Logistic Regression)")
print("=" * 58)
print(f"Accuracy     : {test_acc*100:6.2f}%")
print(f"Macro F1     : {test_f1m*100:6.2f}%")
print(f"Weighted F1  : {test_f1w*100:6.2f}%")
print()
print(f"Reference - always predicting '{LABEL_NAMES[majority]}': {maj_acc*100:.2f}%")
print(f"Lift over that naive baseline: +{(test_acc-maj_acc)*100:.1f} percentage points")
print()
print(classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values()),
                            digits=4, zero_division=0))
'''))

CELLS.append(code('''p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2, 3],
                                            zero_division=0)
per_class = pd.DataFrame({
    "class": list(LABEL_NAMES.values()),
    "precision": p.round(3), "recall": r.round(3), "f1": f.round(3), "support": s,
})
display(per_class)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
            xticklabels=list(LABEL_NAMES.values()),
            yticklabels=list(LABEL_NAMES.values()))
axes[0].set_title(f"Counts (n={len(y_test)})")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

cmn = cm / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cmn, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
            xticklabels=list(LABEL_NAMES.values()),
            yticklabels=list(LABEL_NAMES.values()))
axes[1].set_title("Row-normalised (% of actual class)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

plt.suptitle("Baseline confusion matrix on the held-out test set", fontweight="bold")
plt.tight_layout()
plt.show()
'''))

CELLS.append(md("""**Reading the confusion matrix.** The diagonal is strong for **Negative** only (80% recall).
**Irrelevant** scatters across all four predictions and is caught just 13.6% of the time, despite
high precision (0.82) - the signature of a class the model can recognise but almost never selects.

### 4.5 The imbalance lever

Because the rubric explicitly marks "handling imbalance", we quantify the trade-off rather than
just mentioning it.
"""))

CELLS.append(code('''final_bal = build_pipeline(balanced=True).fit(X_train_lemma, y_train)
y_pred_bal = final_bal.predict(X_test_lemma)

bal_acc = accuracy_score(y_test, y_pred_bal)
bal_f1m = f1_score(y_test, y_pred_bal, average="macro")
bal_f1w = f1_score(y_test, y_pred_bal, average="weighted")
pb, rb, fb, _ = precision_recall_fscore_support(y_test, y_pred_bal, labels=[0, 1, 2, 3],
                                                zero_division=0)

comparison = pd.DataFrame({
    "configuration": ["Plain baseline", "class_weight='balanced'", "Change"],
    "accuracy": [test_acc, bal_acc, bal_acc - test_acc],
    "macro_f1": [test_f1m, bal_f1m, bal_f1m - test_f1m],
    "weighted_f1": [test_f1w, bal_f1w, bal_f1w - test_f1w],
})
display(comparison.style.format({"accuracy": "{:+.4f}", "macro_f1": "{:+.4f}",
                                 "weighted_f1": "{:+.4f}"}))

print("Per-class F1 after balancing:")
print(pd.DataFrame({"class": list(LABEL_NAMES.values()), "F1 plain": f.round(3),
                    "F1 balanced": fb.round(3)}).to_string(index=False))
'''))

CELLS.append(md("""**The trade-off, stated plainly.** Balanced weighting gives up about 1 point of accuracy to gain
about 1.6 points of macro F1, and lifts Irrelevant F1 from 0.234 to 0.312. For a brand-monitoring
system that must not ignore a whole category, that is the right trade - and it is worth 10% of the
final rubric.

### 4.6 What did the model learn?

Because logistic regression is linear, the coefficients are directly readable. This is a
sanity check that the model learned *meaning* rather than an artefact of the split.
"""))

CELLS.append(code('''tfidf = final.named_steps["tfidf"]
clf = final.named_steps["clf"]
features = np.array(tfidf.get_feature_names_out())

fig, axes = plt.subplots(2, 2, figsize=(12.5, 7))
for ax, lab in zip(axes.ravel(), range(4)):
    coefs = clf.coef_[lab]
    top = np.argsort(coefs)[-12:][::-1]
    ax.barh(features[top][::-1], coefs[top][::-1], color=PALETTE[lab],
            alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_title(f"{LABEL_NAMES[lab]} - strongest positive terms")
    ax.set_xlabel("Coefficient")
    ax.tick_params(axis="y", labelsize=8)

plt.suptitle("What the baseline learned (top TF-IDF terms per class)", fontweight="bold")
plt.tight_layout()
plt.show()

print("Top 8 terms per class:")
for lab in range(4):
    idx = np.argsort(clf.coef_[lab])[-8:][::-1]
    print(f"  {LABEL_NAMES[lab]:<11}: {', '.join(features[idx])}")
'''))

# ---------------------------------------------------------------- 5. errors
CELLS.append(md("""---
## 5. Error Analysis

### 5.1 Where the errors are

The baseline misclassifies about half the test rows, but the errors are highly structured rather
than random. Grouping them by (actual, predicted) tells us what to fix.
"""))

CELLS.append(code('''errors = pd.DataFrame({
    "phrase": test["phrase"].astype(str),
    "actual": [LABEL_NAMES[i] for i in y_test],
    "predicted": [LABEL_NAMES[i] for i in y_pred],
    "confidence": y_proba.max(axis=1).round(3),
})
errors["correct"] = errors["actual"] == errors["predicted"]

print(f"Misclassified: {(~errors['correct']).sum()} of {len(errors)} "
      f"({(~errors['correct']).mean()*100:.1f}%)")
print()
print("Most frequent confusion pairs:")
confusions = (errors[~errors["correct"]]
              .groupby(["actual", "predicted"]).size()
              .sort_values(ascending=False).head(8))
for (a, b), n in confusions.items():
    print(f"  {a:<11} -> {b:<11} {n:>3}")
'''))

CELLS.append(code('''print("High-confidence mistakes (the model was wrong AND sure about it):")
print("=" * 100)
hard = errors[~errors["correct"]].sort_values("confidence", ascending=False).head(6)
for _, r in hard.iterrows():
    print(f"[actual={r['actual']:<10} pred={r['predicted']:<10} conf={r['confidence']:.2f}]")
    print(f"   {r['phrase'][:140]}")
'''))

CELLS.append(md("""**Why these particular errors happen**

| Confusion | Count | Cause |
|---|---|---|
| Positive -> Negative | 41 | Sarcasm and backhanded praise: "great, another crash" |
| Neutral -> Negative | 34 | A news sentence about a complaint inherits complaint vocabulary |
| Neutral -> Positive | 29 | Product news shares praise vocabulary ("launch", "new", "win") |
| Irrelevant -> Positive | 24 | With no entity anchor, the model falls back to sentiment |
| Negative -> Positive | 18 | Profanity used affectionately ("this is sick") |
| Positive -> Neutral | 13 | Understated praise ("not bad at all") |

The recurring theme is **context that a bag-of-words model cannot represent**: negation scope,
sarcasm, and - critically - *what the tweet is about*.

### 5.2 The Irrelevant problem is structural, not a tuning problem

We investigated the cause instead of assuming it was a hyper-parameter issue.
"""))

CELLS.append(code('''print("Is 'Irrelevant' separable from the text alone?")
print("=" * 70)

brand_re = brand_regex()
for lab in range(4):
    sub = train.loc[train["sentiment"] == lab, "phrase"].astype(str)
    print(f"{LABEL_NAMES[lab]:<11} n={len(sub):>3}  "
          f"brand mention {sub.str.contains(brand_re, case=False, regex=True).mean()*100:5.1f}%  "
          f"news words {sub.str.contains(r'\\\\b(?:news|report|announce|launch|update)\\\\w*', case=False, regex=True).mean()*100:5.1f}%")

print()
print("Random samples from the 'Irrelevant' class:")
for s in train.loc[train["sentiment"] == 2, "phrase"].astype(str).sample(6, random_state=7):
    print("   -", s[:115])
'''))

CELLS.append(md("""**The structural argument.** "Irrelevant" means *"the message is not about the entity"*. With no
entity column, two annotators given the same tweet could not agree on this label - and we already
found one contradictory pair in the training data.

Look at the samples above: several are perfectly emotional tweets (`Thank you for very
understanding`, `He is so beautiful`) that are simply *about someone else*. Their text carries
sentiment; only the missing entity makes them "irrelevant". A text-only model therefore has
**no information** with which to separate them from ordinary Positive/Negative tweets.

This places a hard ceiling on any model trained on this file, and it is the single most important
item to raise with the professor.

### 5.3 Overfitting check

A learning curve separates two very different problems: *high bias* (model too simple, more data
won't help) and *high variance* (model memorising, regularisation or more data will help).
"""))

CELLS.append(code('''from sklearn.model_selection import learning_curve

sizes, tr_scores, cv_scores = learning_curve(
    build_pipeline(False), X_train_lemma, y_train,
    train_sizes=np.linspace(0.2, 1.0, 5), cv=5,
    scoring="f1_macro", shuffle=True, random_state=RANDOM_STATE,
)

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(sizes, tr_scores.mean(axis=1), "o-", color="#2CA02C", label="Train macro-F1")
ax.plot(sizes, cv_scores.mean(axis=1), "s-", color="#D62728", label="CV macro-F1")
ax.fill_between(sizes, cv_scores.mean(axis=1) - cv_scores.std(axis=1),
                cv_scores.mean(axis=1) + cv_scores.std(axis=1), alpha=0.18, color="#D62728")
ax.set_xlabel("Training examples used")
ax.set_ylabel("Macro F1")
ax.set_title("Learning curve: variance problem, not data starvation", fontweight="bold")
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.show()

print(f"Train F1 at full size : {tr_scores.mean(axis=1)[-1]*100:.1f}%")
print(f"CV    F1 at full size : {cv_scores.mean(axis=1)[-1]*100:.1f}%")
print(f"Gap                   : {(tr_scores.mean(axis=1)[-1]-cv_scores.mean(axis=1)[-1])*100:.1f} pp")
'''))

CELLS.append(md("""**Interpretation.** The training F1 reaches ~99% while cross-validation F1 is only ~37%. A gap that
large means the classifier is **memorising** training tweets rather than learning generalisable
rules - a variance problem. The curve has not plateaued, so more data would also help, but
regularisation, feature selection and pretrained embeddings are the higher-leverage responses.

This is exactly why Weeks 11-13 move to dense contextual embeddings: with 592 short tweets there is
simply not enough lexical evidence for a bag-of-words model to generalise.
"""))

# ---------------------------------------------------------------- 6. findings
CELLS.append(md("""---
## 6. Findings and Open Questions for the Consultation

### 6.1 Summary of what we established

| # | Finding | Evidence | Implication |
|---|---|---|---|
| 1 | The delivered data has no `entity` column, although the task is defined as entity-level | Section 1.3 | Label 2 is partly unlearnable; macro F1 has a structural ceiling |
| 2 | Imbalance is mild (1.65:1) but label overlap is severe | Sections 2.1, 5.1 | Accuracy is meaningful; the real difficulty is ambiguity, not skew |
| 3 | 15.8% of rows had double-encoded UTF-8, plus 4.4% with `<unk>` | Section 1.5 | Objective corruption; repaired, and mandatory before BERT |
| 4 | Cleaning is neutral for accuracy but necessary for correctness | Section 4.3 | Do not over-claim; justify by validity and downstream needs |
| 5 | Class-weighted training is the biggest single lever (+6.6 pp CV macro F1) | Sections 4.3, 4.5 | Concrete starting point for Phase 3iv |

### 6.2 Benchmark to carry forward

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Majority-class reference | 30.2% | - | - |
| **TF-IDF + Logistic Regression (baseline)** | **49.6%** | **44.0%** | **46.7%** |
| + `class_weight='balanced'` | 48.6% | 45.6% | 47.6% |

These are the numbers every Phase 3ii model must beat.

### 6.3 Questions for the professor

1. **How should label 2 be treated** given that the entity column is missing - as a genuine class,
   or as out-of-scope noise to exclude from scoring?
2. **May the provided `testdata.csv` be used for final model comparison**, or should we carve a
   validation split out of train and reserve test for a single final evaluation?
3. **Which metric is primary for marking** - accuracy or macro F1? Our two candidate baselines
   differ on which they favour.

### 6.4 Plan for Weeks 11-13

With five group members, N = 5 analytics models each paired with a different embedding:

| # | Embedding | Analytics model | Rationale |
|---|---|---|---|
| 1 | GloVe (pre-trained) | Gradient Boosting (XGBoost / LightGBM) | strong learner on dense vectors |
| 2 | Word2Vec (trained from scratch) | Random Forest | satisfies the "from scratch" bonus |
| 3 | FastText (subword) | SVM | subword units handle misspellings and elongation |
| 4 | BERT (fine-tuned) | Neural classifier head | highest expected ceiling; captures negation and order |
| 5 | Sentence-BERT | kNN / MLP | non-parametric contrast to the discriminative models |

The five are deliberately from **different model families** - the brief penalises a set of near
identical tree models for lack of diversity.
"""))

CELLS.append(md("""---
## 7. Reproducibility

Everything in this notebook is deterministic. To regenerate the written report and every figure
from the raw CSVs:

```bash
python src/run_pipeline.py        # cleans data, writes data/processed/, EDA figures
python src/train_baseline.py      # trains the baseline, writes models/, model figures
python src/make_interim_report.py # rebuilds the .docx from the JSON artefacts
```

**Design note:** the report generator reads its numbers from
`models/baseline_metrics.json` and `data/processed/eda_summary.json` rather than from
hand-typed text. If the pipeline is re-run on refreshed data, the document regenerates with the new
values and cannot silently disagree with the code.

### Artefact index

| Artefact | Location |
|---|---|
| Shared cleaning functions | `src/text_clean.py` |
| Cleaning pipeline + EDA | `src/run_pipeline.py` |
| Baseline training | `src/train_baseline.py` |
| Cleaned data | `data/processed/train_clean.csv`, `test_clean.csv` |
| Cleaning audit log | `data/processed/cleaning_log.json` |
| Fitted model | `models/baseline_tfidf_logreg.joblib` |
| All metrics | `models/baseline_metrics.json` |
| Figures | `reports/figures/` |
| Metric tables | `reports/tables/` |
"""))

CELLS.append(md("""---
*End of interim technical notebook - Group 8 ChillGuy - DSAI4205 Big Data Analytics (2026/27 S1)*
"""))


def build() -> Path:
    # assign stable cell ids (nbformat 4.5+ requires them; avoids validator warnings)
    for i, cell in enumerate(CELLS):
        cell.setdefault("id", f"cell-{i:03d}")

    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.7",
                              "mimetype": "text/x-python", "file_extension": ".py",
                              "pygments_lexer": "ipython3",
                              "codemirror_mode": {"name": "ipython", "version": 3},
                              "nbconvert_exporter": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return OUT


if __name__ == "__main__":
    p = build()
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"written -> {p}")
    print(f"cells   -> {len(CELLS)} total ({n_code} code, {len(CELLS)-n_code} markdown)")
    print(f"size    -> {p.stat().st_size/1024:.1f} KB")
