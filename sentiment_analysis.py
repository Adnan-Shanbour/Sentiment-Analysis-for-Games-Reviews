"""
Sentiment Analysis Pipeline
(converted from sentiment_analysis.ipynb)

Feature Representations : TF-IDF (unigrams + bigrams) · Word2Vec (averaged)
Classifiers            : Logistic Regression · Linear SVM · Naive Bayes

The notebook's side-effecting helpers (print / plt.show) have been refactored
into pure functions that return structured results so they can be consumed by
a GUI (see app.py).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection         import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.svm                     import LinearSVC
from sklearn.naive_bayes             import MultinomialNB, GaussianNB
from sklearn.metrics                 import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing           import LabelEncoder
from gensim.models                   import Word2Vec


# ═════════════════════════════════════════════════════════════════════
# Data preparation
# ═════════════════════════════════════════════════════════════════════
def prepare_data(df, test_size=0.20, random_state=42):
    """Clean, label-encode and split the dataframe 80/20 (stratified).

    The input dataframe must contain at least these columns:
        cleaned_text, tokens, label
    The `tokens` column may be either a real list or a pipe-separated string.
    """
    df = df.dropna(subset=["cleaned_text", "label"]).copy()
    df["cleaned_text"] = df["cleaned_text"].astype(str)

    # Normalise tokens column to list[str]
    if "tokens" in df.columns:
        first = df["tokens"].iloc[0] if len(df) else ""
        if isinstance(first, str):
            df["tokens"] = df["tokens"].astype(str).apply(
                lambda s: [t.strip() for t in s.split("|") if t.strip()]
            )
    else:
        df["tokens"] = df["cleaned_text"].apply(lambda s: s.split())

    labels = sorted(df["label"].unique().tolist())
    le = LabelEncoder()
    le.fit(labels)
    y      = le.transform(df["label"])
    X_text = df["cleaned_text"].values

    all_indices = np.arange(len(df))
    train_pos, test_pos = train_test_split(
        all_indices, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "df":           df.reset_index(drop=True),
        "labels":       labels,
        "le":           le,
        "y":            y,
        "X_text":       X_text,
        "train_pos":    train_pos,
        "test_pos":     test_pos,
        "X_train_text": X_text[train_pos],
        "X_test_text":  X_text[test_pos],
        "y_train":      y[train_pos],
        "y_test":       y[test_pos],
    }


# ═════════════════════════════════════════════════════════════════════
# Feature representations
# ═════════════════════════════════════════════════════════════════════
def build_tfidf(X_train_text, X_test_text,
                ngram_range=(1, 2), max_features=20_000,
                sublinear_tf=True, min_df=2):
    """Fit a TF-IDF vectorizer on train, transform both splits."""
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf  = tfidf.transform(X_test_text)
    return tfidf, X_train_tfidf, X_test_tfidf


def build_word2vec(df, train_pos, test_pos,
                   vector_size=100, window=5, min_count=1,
                   workers=4, seed=42, epochs=10):
    """Train Word2Vec on the full corpus, return averaged document vectors."""
    all_token_lists = df["tokens"].tolist()

    w2v_model = Word2Vec(
        sentences=all_token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
        epochs=epochs,
    )

    train_tokens = df["tokens"].iloc[train_pos].tolist()
    test_tokens  = df["tokens"].iloc[test_pos].tolist()

    X_train_w2v = _tokens_to_matrix(train_tokens, w2v_model, vector_size)
    X_test_w2v  = _tokens_to_matrix(test_tokens,  w2v_model, vector_size)

    return w2v_model, X_train_w2v, X_test_w2v


def _tokens_to_matrix(token_lists, w2v_model, dim):
    vecs = []
    for tokens in token_lists:
        valid = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
        vecs.append(np.mean(valid, axis=0) if valid else np.zeros(dim))
    return np.array(vecs)


def tokens_to_vector(tokens, w2v_model, dim=100):
    """Convert a single list of tokens into an averaged W2V vector."""
    valid = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if valid:
        return np.mean(valid, axis=0)
    return np.zeros(dim)


# ═════════════════════════════════════════════════════════════════════
# Training & evaluation
# ═════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_train, X_test, y_train, y_test,
                   labels, model_name="Model"):
    """Fit `model` and return a dict of metrics + the trained estimator."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score   (y_test, y_pred, average="macro", zero_division=0)
    f1m  = f1_score       (y_test, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    report_text = classification_report(
        y_test, y_pred, target_names=labels, zero_division=0
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=labels, zero_division=0, output_dict=True
    )

    return {
        "name":        model_name,
        "clf":         model,
        "y_pred":      y_pred,
        "acc":         acc,
        "prec":        prec,
        "rec":         rec,
        "f1_macro":    f1m,
        "cm":          cm,
        "report":      report_dict,
        "report_text": report_text,
    }


def tune_model(factory, param_grid, X_train, X_test, y_train, y_test,
               labels, model_name="Model", cv=3, scoring="f1_macro"):
    """Run GridSearchCV with the estimator returned by `factory()`.

    Returns the same structure as `evaluate_model` plus the best params / cv score.
    """
    grid = GridSearchCV(
        factory(), param_grid,
        cv=cv, scoring=scoring, n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)

    result = evaluate_model(
        grid.best_estimator_, X_train, X_test, y_train, y_test,
        labels, model_name=f"{model_name} (tuned)",
    )
    result["best_params"] = grid.best_params_
    result["cv_score"]    = grid.best_score_
    return result


# ═════════════════════════════════════════════════════════════════════
# Top features (linear models + TF-IDF)
# ═════════════════════════════════════════════════════════════════════
def get_top_features(model, vectorizer, labels, top_n=15):
    """Return the top-N most positive / negative features per class.

    Works with any estimator that exposes a `coef_` attribute
    (Logistic Regression, Linear SVM, …).  Returns None otherwise.
    """
    if not hasattr(model, "coef_"):
        return None

    feature_names = np.array(vectorizer.get_feature_names_out())
    results = {}

    for i, label in enumerate(labels):
        coefs = model.coef_[i] if model.coef_.ndim > 1 else model.coef_[0]
        top_pos = feature_names[np.argsort(coefs)[-top_n:]][::-1]
        top_neg = feature_names[np.argsort(coefs)[:top_n]]
        results[label] = {
            "positive": top_pos.tolist(),
            "negative": top_neg.tolist(),
        }
    return results
