"""
Text cleaning utilities for the Twitter entity-level sentiment dataset.

Design notes
------------
* Every function is pure and returns new objects, so the pipeline is reproducible
  and each step can be ablated / unit-tested.
* Cleaning is *lossy*, so we always keep `Phrase` (raw) next to `text` (clean).
  The raw column is required because label 2 ("Irrelevant") can only be judged
  against the original wording and because the interim report must show
  before/after examples.
* `strip_mentions` / `strip_urls` are optional because @brand handles and URLs
  carry real signal (complaints are usually addressed to a support handle).
  The baseline is evaluated with them REMOVED (conservative, avoids the model
  memorising handles) and the effect is reported in the interim report.
"""
from __future__ import annotations

import re
import unicodedata

import ftfy

# ---------------------------------------------------------------- regexes
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\s?\w+")
HASHTAG_RE = re.compile(r"#\s?(\w+)")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^<>]{0,30}>")
UNK_RE = re.compile(r"<\s*unk\s*>", re.IGNORECASE)
# words that were glued to <unk> during the original tokenisation, e.g. "rework<unk>"
GLUED_UNK_RE = re.compile(r"(?<=[A-Za-z])<\s*unk\s*>|<\s*unk\s*>(?=[A-Za-z])", re.IGNORECASE)
HTML_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|apos|nbsp|#\d+);")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_KEEP_RE = re.compile(r"[a-z']+")
# Twitter-style elongation: "soooooo" -> "soo", "!!!!!!" -> "!!"
ELONGATION_RE = re.compile(r"([a-z])\1{2,}")
PUNCT_REPEAT_RE = re.compile(r"([!?.,])\1+")

# Negations must survive stop-word removal or sentiment flips meaning.
NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither",
             "nor", "cannot", "can't", "don't", "doesn't", "didn't", "isn't",
             "wasn't", "aren't", "weren't", "won't", "wouldn't", "shouldn't",
             "couldn't", "ain't", "hasn't", "haven't", "hadn't"}

# Contraction expansion applied BEFORE punctuation removal so that
# "can't" -> "can not" instead of "cant".
CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
    "'m": " am", "'s": " is",
}


def fix_encoding(text: str) -> str:
    """Repair mojibake (â€™ -> ’, ðŸ¤¦ -> emoji) and normalise unicode.

    The scraper that produced this dataset double-encoded UTF-8 as cp1252,
    which is why 94/596 train rows contain 'Ã'/'â'/'ð'. ftfy reverses exactly
    that class of corruption; NFKC then collapses full-width and ligature forms.
    """
    if not isinstance(text, str):
        return ""
    text = ftfy.fix_text(text)
    return unicodedata.normalize("NFKC", text)


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    return MENTION_RE.sub(" ", text)


def normalise_hashtags(text: str) -> str:
    """'#GamingIsLife' -> 'Gaming Is Life' (keep the words, drop the marker)."""
    return HASHTAG_RE.sub(lambda m: " " + m.group(1) + " ", text)


def remove_html(text: str) -> str:
    text = HTML_ENTITY_RE.sub(" ", text)
    return HTML_TAG_RE.sub(" ", text)


def remove_unk(text: str) -> str:
    """Drop the corpus's <unk> placeholder without welding neighbouring words."""
    text = GLUED_UNK_RE.sub(" ", text)
    return UNK_RE.sub(" ", text)


def expand_contractions(text: str) -> str:
    for short, full in CONTRACTIONS.items():
        text = text.replace(short, full)
    return text


def collapse_repeats(text: str) -> str:
    """'sooooo goooood!!!!' -> 'soo good!!'.

    Keeps a double letter/char so 'good' is not damaged but 'goooood' becomes
    'good'-adjacent rather than a unique out-of-vocabulary token.
    """
    text = ELONGATION_RE.sub(r"\1\1", text)
    return PUNCT_REPEAT_RE.sub(r"\1\1", text)


def basic_clean(text: str, *, drop_urls: bool = True, drop_mentions: bool = True) -> str:
    """Full normalisation chain -> lowercase, punctuation-free, single-spaced."""
    text = fix_encoding(text)
    text = remove_html(text)
    text = remove_unk(text)
    text = normalise_hashtags(text)
    if drop_urls:
        text = remove_urls(text)
    if drop_mentions:
        text = remove_mentions(text)
    text = text.lower()
    text = expand_contractions(text)
    text = collapse_repeats(text)
    # keep letters, apostrophes and spaces only
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"(?<![a-z])'+|'+(?![a-z])", " ", text)  # stray quotes
    return WHITESPACE_RE.sub(" ", text).strip()


def tokenise(text: str) -> list[str]:
    return TOKEN_KEEP_RE.findall(text)


def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    """Standard stop-word removal that protects negation words."""
    return [t for t in tokens if t not in stopwords or t in NEGATIONS]


def lemmatise(tokens: list[str], lemmatiser) -> list[str]:
    """Lemmatise only alphabetic tokens with len>2 (spaCy handles the rest)."""
    out = []
    for t in tokens:
        if len(t) > 2 and t.isalpha():
            lemma = lemmatiser(t)
            out.append(lemma if lemma else t)
        else:
            out.append(t)
    return out


def _wordnet_lemma(word: str) -> str:
    """Verb-first, then noun fallback: 'crashes'->'crash', 'heroes'->'hero'."""
    from nltk.stem import WordNetLemmatizer
    wn = WordNetLemmatizer()
    v = wn.lemmatize(word, "v")
    return wn.lemmatize(v, "n")


def _get_lemmatiser():
    """Return a cached, NLTK-backed lemmatiser (WordNet, verb + noun pass)."""
    global _LEMMATISER
    if _LEMMATISER is None:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            wn = WordNetLemmatizer()

            def _fn(w: str) -> str:
                return wn.lemmatize(wn.lemmatize(w, "v"), "n")

            # make sure the WordNet corpus is present; download quietly if not
            try:
                wn.lemmatize("testing", "v")
            except LookupError:  # pragma: no cover
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            _LEMMATISER = _fn
        except Exception:  # pragma: no cover - graceful degradation
            _LEMMATISER = lambda w: w  # noqa: E731
    return _LEMMATISER


_LEMMATISER = None


def token_count(text: str) -> int:
    return len(WHITESPACE_RE.split(text.strip())) if text and text.strip() else 0
