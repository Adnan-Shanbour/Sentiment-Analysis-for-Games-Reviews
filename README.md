# Game Review Sentiment Analysis

A complete NLP pipeline for three-class sentiment classification of mobile game reviews, built for **CSAI 452 – Natural Language Processing**. The project covers data collection, preprocessing, classical ML experimentation, and an interactive Streamlit demo app.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Feature Extraction](#2-feature-extraction)
  - [3. Models & Hyperparameter Tuning](#3-models--hyperparameter-tuning)
- [Results](#results)
- [Interactive App](#interactive-app)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

Mobile game reviews are noisy, short, and emotionally expressive — making them a challenging benchmark for sentiment analysis. This project classifies reviews as **Positive**, **Negative**, or **Neutral** using a head-to-head comparison of six model configurations across two feature representations.

**Key highlights:**
- 8-stage text preprocessing pipeline with explicit negation preservation
- Two feature representations: TF-IDF (sparse) vs. Word2Vec (dense)
- Three classifiers: Logistic Regression, Linear SVM, Naive Bayes
- Baseline vs. GridSearchCV hyperparameter tuning for all combinations
- Streamlit GUI for interactive training, evaluation, and live prediction

---

## Project Structure

```
project/
├── app.py                          # Streamlit web application
├── preprocessing.py                # Preprocessing module (importable)
├── preprocessing.ipynb             # Preprocessing notebook with EDA
├── sentiment_analysis.py           # Training & evaluation module (importable)
├── sentiment_analysis.ipynb        # Experiment notebook (baseline + tuned)
├── sentiment_analysis_with_bert.ipynb  # BERT-based extension
├── scrape_playstore_reviews.py     # Google Play Store review scraper
├── reviews_dataset.csv             # Raw dataset (1,110 reviews)
└── reviews_preprocessed.csv       # Cleaned dataset (output of preprocessing)
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Google Play Store (scraped) |
| Total reviews | 1,110 |
| Class distribution | 370 Positive / 370 Negative / 370 Neutral (perfectly balanced) |
| Columns | `review_id`, `source`, `product_category`, `review_text`, `rating`, `label` |
| Train / Test split | 80 / 20 stratified (888 train, 222 test) |

The dataset was scraped from the Play Store using `scrape_playstore_reviews.py` and manually labelled to ensure a balanced three-class distribution.

---

## Pipeline

### 1. Preprocessing

Eight sequential stages applied to every review:

| Step | Operation |
|---|---|
| 1 | Lowercase |
| 2 | Remove URLs (HTTP/HTTPS/www) |
| 3 | Strip HTML tags and entities |
| 4 | Convert emojis to text via `emoji.demojize()` |
| 5 | Remove punctuation and digits |
| 6 | Tokenize with NLTK `word_tokenize` |
| 7 | Remove stopwords — **negations preserved** (`not`, `no`, `never`, `nor`, `neither`, `hardly`, `barely`, `scarcely`) |
| 8 | Lemmatize with NLTK `WordNetLemmatizer` (verb → noun fallback) |

> **Design note:** Negations are explicitly kept because flipping sentiment polarity (*"not good"* → *"good"*) would silently corrupt labels in a bag-of-words model.

Output columns added to the CSV: `cleaned_text`, `tokens`, `token_count`.

---

### 2. Feature Extraction

**TF-IDF**
- Unigrams + bigrams (`ngram_range=(1,2)`)
- 20,000 max features, sublinear TF scaling (`log(tf) + 1`), minimum document frequency of 2
- Resulting vocabulary: **5,013 features**

**Word2Vec**
- Trained on the full corpus (no label leakage)
- `vector_size=100`, `window=5`, `min_count=1`, `epochs=10`
- Vocabulary: **4,046 words**
- Document vector: mean of all in-vocabulary word vectors

---

### 3. Models & Hyperparameter Tuning

| Classifier | TF-IDF Search Space | Word2Vec Search Space |
|---|---|---|
| Logistic Regression | `C ∈ {0.1, 1, 10}` | `C ∈ {0.1, 1, 10}` |
| Linear SVM | `C ∈ {0.1, 1, 10}` | `C ∈ {0.1, 1, 10}` |
| Naive Bayes | `MultinomialNB α ∈ {0.1, 0.5, 1.0, 5.0}` | `GaussianNB var_smoothing ∈ {1e-9, 1e-8, 1e-7}` |

All tuning uses 5-fold stratified cross-validation via `GridSearchCV`.

---

## Results

All metrics are macro-averaged over the three classes.

| Configuration | Accuracy | Precision | Recall | F1-macro |
|---|---|---|---|---|
| **NB + TF-IDF (tuned)** | **0.6306** | **0.6293** | **0.6306** | **0.6285** |
| NB + TF-IDF (baseline) | 0.6126 | 0.6172 | 0.6126 | 0.6143 |
| LR + TF-IDF (tuned) | 0.6171 | 0.6129 | 0.6171 | 0.6098 |
| LR + TF-IDF (baseline) | 0.5856 | 0.5823 | 0.5856 | 0.5799 |
| SVM + TF-IDF (tuned) | 0.5946 | 0.5896 | 0.5946 | 0.5841 |
| SVM + TF-IDF (baseline) | 0.5495 | 0.5400 | 0.5495 | 0.5424 |
| SVM + Word2Vec (tuned) | 0.5541 | 0.5506 | 0.5541 | 0.5378 |
| LR + Word2Vec (tuned) | 0.5180 | 0.5152 | 0.5180 | 0.5044 |
| LR + Word2Vec (baseline) | 0.4910 | 0.5019 | 0.4910 | 0.4860 |
| SVM + Word2Vec (baseline) | 0.5000 | 0.5020 | 0.5000 | 0.4849 |
| NB + Word2Vec (tuned) | 0.3604 | 0.3593 | 0.3604 | 0.3556 |
| NB + Word2Vec (baseline) | 0.3604 | 0.3593 | 0.3604 | 0.3556 |

**Best model: Multinomial Naive Bayes + TF-IDF (tuned) — F1-macro 0.6285**

**Key takeaways:**
- TF-IDF consistently outperforms Word2Vec across all three classifiers. Sparse, discrete term counts are a better fit for short bag-of-words sentiment than averaged dense embeddings on this dataset size.
- Naive Bayes pairs especially well with TF-IDF's sparse representation; it degrades with Word2Vec's continuous features (GaussianNB).
- The neutral class is the hardest to classify — reviews with mixed sentiment and mid-range ratings frequently cross the positive/neutral boundary.

**Most predictive features (NB + TF-IDF):**

| Class | Top features |
|---|---|
| Positive | best, graphic, fun, love, mobile, amaze, character, world, great, excite |
| Negative | not, worst, worse, face, win, bot, money, lose, spend, trash, frustrate |
| Neutral | annoy, thing, time, something, need, sometimes, crash, connection, app, ad |

---

## Interactive App

`app.py` is a Streamlit application with three tabs:

| Tab | Functionality |
|---|---|
| **Dataset** | Load CSV, run preprocessing, view class distribution and dataset statistics |
| **Train Models** | Select feature type and classifier, toggle hyperparameter tuning, view confusion matrices, classification reports, and misclassified examples |
| **Predict** | Enter any review text, pick a trained model, get a sentiment prediction with class probabilities |

The UI auto-detects light/dark mode and uses theme-aware styling.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Adnan-Shanbour/game-review-sentiment.git
cd game-review-sentiment

# Install dependencies
pip install streamlit pandas numpy scikit-learn gensim nltk emoji matplotlib seaborn

# Download required NLTK data (run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

> Python 3.9+ recommended.

---

## Usage

There are two ways to explore this project — pick whichever fits your workflow:

---

### Option A — Streamlit GUI (recommended for a quick overview)

Run the app and do everything through the browser interface: load the dataset, preprocess, train models, compare results, and predict on custom text — all in one place.

```bash
streamlit run app.py
```

---

### Option B — Notebooks (recommended for step-by-step inspection)

Run the notebooks in order to see every stage in detail — intermediate outputs, plots, and metrics are displayed inline as each cell executes.

1. `preprocessing.ipynb` — Cleans `reviews_dataset.csv` → produces `reviews_preprocessed.csv`
2. `sentiment_analysis.ipynb` — Trains and evaluates all 12 model configurations, baseline and tuned
3. `sentiment_analysis_with_bert.ipynb` — BERT-based extension (optional)

---

**Scrape new reviews:**
```bash
python scrape_playstore_reviews.py
```

---

## Acknowledgements

Built as the course project for **CSAI 452 – Natural Language Processing**.  
Dataset collected from the Google Play Store.
