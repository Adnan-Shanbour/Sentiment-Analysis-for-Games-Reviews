"""
Reviews Preprocessing Pipeline
(converted from preprocessing.ipynb)

Steps:
    1. Lowercasing
    2. Remove URLs
    3. Remove HTML tags
    4. Convert emojis -> words
    5. Remove punctuation & extra spaces
    6. Tokenization
    7. Stopword removal (keep negations — they affect sentiment)
    8. Lemmatization
"""

import re
import nltk
import emoji

# Download required NLTK data (runs once, cached afterwards)
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ─── Stopwords & Lemmatizer ───────────────────────────────────────────
NEGATIONS = {
    "not", "no", "never", "nor", "neither", "nothing",
    "nobody", "nowhere", "hardly", "barely", "scarcely",
}

STOP_WORDS = set(stopwords.words("english")) - NEGATIONS

lemmatizer = WordNetLemmatizer()


# ─── Cleaning Functions ───────────────────────────────────────────────
def lowercase(text):
    return text.lower()


def remove_urls(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text


def remove_html(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    return text


def convert_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))


def remove_punctuation(text):
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"'\s|'\s*$", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def lemmatize(tokens):
    lemmas = [lemmatizer.lemmatize(t, pos="v") for t in tokens]
    lemmas = [lemmatizer.lemmatize(t, pos="n") for t in lemmas]
    return lemmas


# ─── Full Pipeline ────────────────────────────────────────────────────
def preprocess(text):
    """Run the full cleaning pipeline on a single review string.

    Returns:
        dict with keys: cleaned_text (str), tokens (list[str]), token_count (int)
    """
    if text is None:
        text = ""
    text = str(text)

    s1 = lowercase(text)
    s2 = remove_urls(s1)
    s3 = remove_html(s2)
    s4 = convert_emojis(s3)
    s5 = remove_punctuation(s4)
    s6 = tokenize(s5)
    s7 = remove_stopwords(s6)
    s8 = lemmatize(s7)

    return {
        "cleaned_text": " ".join(s8),
        "tokens": s8,
        "token_count": len(s8),
    }
