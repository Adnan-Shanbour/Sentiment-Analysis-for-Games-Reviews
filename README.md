<div align="center">

# 🎮 Game Review Sentiment Analysis

**Three-class NLP sentiment classification for mobile game reviews**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=flat-square)](https://www.nltk.org/)
[![Course](https://img.shields.io/badge/CSAI%20452-Natural%20Language%20Processing-purple?style=flat-square)](.)

<br/>

*From raw Play Store reviews to a fully interactive prediction app, end-to-end.*

</div>

<br/>

## 📌 Overview

Mobile game reviews are noisy, short, and emotionally charged, making them a compelling benchmark for sentiment analysis. This project builds a complete pipeline that classifies reviews as **Positive 🟢**, **Negative 🔴**, or **Neutral 🟡** using classical ML methods.

<div align="center">

| 🗂️ 1,110 Reviews | ⚙️ 8-Stage Pipeline | 🤖 12 Model Configs | 🏆 63% F1-macro |
|:---:|:---:|:---:|:---:|
| Balanced 3-class dataset | End-to-end preprocessing | Baseline + tuned | Best: NB + TF-IDF |

</div>

**What's inside:**
- Scraped, cleaned, and labelled Google Play Store reviews
- 8-stage preprocessing pipeline with deliberate negation preservation
- Two feature representations: **TF-IDF** (sparse) vs. **Word2Vec** (dense)
- Three classifiers: Logistic Regression, Linear SVM, Naive Bayes
- Baseline vs. GridSearchCV tuning across all combinations
- Interactive **Streamlit app** for training, evaluation, and live prediction

<br/>

## 📁 Project Structure

```
project/
├── 🖥️  app.py                              # Streamlit web application
│
├── 📓  preprocessing.ipynb                 # Preprocessing notebook (step-by-step + EDA)
├── 🐍  preprocessing.py                    # Preprocessing module (importable)
│
├── 📓  sentiment_analysis.ipynb            # Experiment notebook (baseline + tuned results)
├── 🐍  sentiment_analysis.py               # Training & evaluation module (importable)
│
├── 📓  sentiment_analysis_with_bert.ipynb  # BERT-based extension
├── 🐍  scrape_playstore_reviews.py         # Google Play Store scraper
│
├── 📊  reviews_dataset.csv                 # Raw dataset  (1,110 reviews)
└── 📊  reviews_preprocessed.csv           # Cleaned dataset (pipeline output)
```

<br/>

## 📊 Dataset

<div align="center">

| Property | Details |
|:---|:---|
| **Source** | Google Play Store (scraped) |
| **Total reviews** | 1,110 |
| **Class balance** | 370 Positive · 370 Negative · 370 Neutral *(perfectly balanced)* |
| **Columns** | `review_id`, `source`, `product_category`, `review_text`, `rating`, `label` |
| **Split** | 80 / 20 stratified: 888 train · 222 test |

</div>

Reviews were scraped with `scrape_playstore_reviews.py` and labelled to maintain a perfectly balanced three-class distribution.

<br/>

## ⚙️ Pipeline

### 🧹 Step 1: Preprocessing

Eight sequential cleaning stages applied to every review:

| # | Stage | Operation |
|:---:|:---|:---|
| 1 | **Lowercase** | Normalize all characters to lowercase |
| 2 | **Remove URLs** | Strip HTTP/HTTPS and www links |
| 3 | **Strip HTML** | Remove tags and entities (e.g. `&amp;`) |
| 4 | **Emoji to Text** | Convert emojis via `emoji.demojize()` |
| 5 | **Clean punctuation** | Remove punctuation and digits, collapse whitespace |
| 6 | **Tokenize** | Split with NLTK `word_tokenize` |
| 7 | **Remove stopwords** | Drop common words, **negations kept** ⚠️ |
| 8 | **Lemmatize** | `WordNetLemmatizer` (verb to noun fallback) |

> **Why keep negations?** Words like *not*, *never*, *hardly*, *barely* directly flip sentiment polarity. Removing them would silently corrupt labels in a bag-of-words model, e.g. *"not good"* becomes *"good"*.

Output columns appended to the CSV: `cleaned_text` · `tokens` · `token_count`

### 🔢 Step 2: Feature Extraction

<table>
<tr>
<th>TF-IDF</th>
<th>Word2Vec</th>
</tr>
<tr>
<td>

- Unigrams + bigrams (`ngram_range=(1,2)`)
- 20,000 max features
- Sublinear TF scaling: `log(tf) + 1`
- Min document frequency: 2
- **Vocabulary: 5,013 features**

</td>
<td>

- Trained on full corpus (no label leakage)
- `vector_size=100`, `window=5`, `min_count=1`
- 10 training epochs
- Document vector = mean of word vectors
- **Vocabulary: 4,046 words**

</td>
</tr>
</table>

### 🤖 Step 3: Models & Hyperparameter Tuning

| Classifier | TF-IDF Search Space | Word2Vec Search Space |
|:---|:---|:---|
| **Logistic Regression** | `C ∈ {0.1, 1, 10}` | `C ∈ {0.1, 1, 10}` |
| **Linear SVM** | `C ∈ {0.1, 1, 10}` | `C ∈ {0.1, 1, 10}` |
| **Naive Bayes** | `MultinomialNB α ∈ {0.1, 0.5, 1.0, 5.0}` | `GaussianNB var_smoothing ∈ {1e-9, 1e-8, 1e-7}` |

All tuning uses **5-fold stratified cross-validation** via `GridSearchCV`.

<br/>

## 🏆 Results

> All metrics are macro-averaged over the three classes.

| Configuration | Accuracy | Precision | Recall | F1-macro |
|:---|:---:|:---:|:---:|:---:|
| 🥇 **NB + TF-IDF (tuned)** | **0.6306** | **0.6293** | **0.6306** | **0.6285** |
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

<details>
<summary><b>Key takeaways</b></summary>
<br/>

- **TF-IDF beats Word2Vec** across all classifiers. Sparse, discrete term counts suit short reviews better than averaged dense embeddings at this dataset size.
- **Naive Bayes thrives with TF-IDF's** sparse representation but degrades with Word2Vec's continuous features (GaussianNB).
- **Neutral is the hardest class:** mixed-sentiment reviews with mid-range ratings (3-4 stars) frequently cross the positive/neutral boundary.
- **Common failure modes:** sarcasm, rating-label mismatch, and out-of-vocabulary words producing zero Word2Vec vectors.

</details>

<details>
<summary><b>Most predictive features: NB + TF-IDF</b></summary>
<br/>

| Sentiment | Top keywords |
|:---:|:---|
| 🟢 Positive | `best` · `graphic` · `fun` · `love` · `mobile` · `amaze` · `character` · `world` · `great` · `excite` |
| 🔴 Negative | `not` · `worst` · `worse` · `face` · `win` · `bot` · `money` · `lose` · `spend` · `trash` · `frustrate` |
| 🟡 Neutral | `annoy` · `thing` · `time` · `something` · `need` · `sometimes` · `crash` · `connection` · `app` · `ad` |

</details>

<br/>

## 🔍 Why Don't the Models Score Higher?

63% F1-macro on a balanced 3-class problem is a meaningful result, but understanding the ceiling matters. The limitations come from several compounding factors, ranked by impact:

**1. 🎯 The neutral class is inherently ambiguous** *(biggest culprit)*

Drawing the line between neutral and mildly positive/negative is subjective. A review like *"graphics are great but the ads ruin it"* is simultaneously positive and negative. Wherever the labeller drew that boundary, another person would draw it differently. That inconsistency is baked into the ground truth, and no model can reliably learn a boundary that isn't consistently defined.

**2. 📉 888 training samples is genuinely small for a 3-class problem**

TF-IDF produced 5,013 features against only 888 training rows, which means more features than samples. The models don't see enough examples to reliably learn which feature combinations distinguish each class, especially near the decision boundaries.

**3. 🧮 Word2Vec was trained from scratch on a tiny corpus**

This explains almost entirely why Word2Vec underperforms TF-IDF so badly. Word2Vec needs millions of sentences to learn meaningful embeddings. Training it on 1,110 short reviews produces nearly random vectors. Using pre-trained embeddings (GloVe, fastText) would have been a much fairer comparison and would likely close the gap significantly.

**4. 📝 Short review length contributes at the margins**

Short texts create sparse TF-IDF vectors and unreliable Word2Vec averages, making individual predictions noisier. That said, TF-IDF still reached 63%, so the signal is there. Length makes the problem harder but isn't the root cause.

**5. 🧠 Classical bag-of-words models can't capture context or sarcasm**

*"Oh great, another pay-to-win update"* — a TF-IDF model sees `great` and leans positive. Without word order or contextual understanding, these cases are unrecoverable with classical ML. BERT and similar models handle this because they read the full sentence as a unit rather than a bag of independent tokens.

> **Highest-leverage improvements:**
> - Use a pre-trained contextual model *(BERT notebook already included)*, which addresses points 3, 4, and 5 at once
> - Collect more labelled data, especially for the neutral class
> - Consider collapsing to binary classification (positive vs. negative): removing the ambiguous neutral class would likely push accuracy above 80%

<br/>

## 🖥️ Interactive App

`app.py` is a Streamlit application with three tabs: load your data, train models, and predict in real time, all in the browser.

| Tab | What you can do |
|:---|:---|
| **📂 Dataset** | Load the CSV, run the full preprocessing pipeline, explore class distribution and dataset statistics |
| **🧠 Train Models** | Choose a feature type and classifier, toggle hyperparameter tuning, compare confusion matrices, classification reports, and misclassified examples |
| **🔮 Predict** | Type any review, select a trained model, get a sentiment label with confidence scores |

The UI auto-detects **light / dark mode** and applies theme-aware styling throughout.

<br/>

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/Adnan-Shanbour/game-review-sentiment.git
cd game-review-sentiment

# 2. Install dependencies
pip install streamlit pandas numpy scikit-learn gensim nltk emoji matplotlib seaborn

# 3. Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

> Requires **Python 3.9+**

<br/>

## 🧭 Usage

There are two ways to explore this project, pick whichever fits your workflow:

### 🖥️ Option A: Streamlit GUI *(recommended for a quick overview)*

Run the app and do everything through the browser: load the dataset, preprocess, train, compare models, and predict on custom text, all in one place.

```bash
streamlit run app.py
```

### 📓 Option B: Notebooks *(recommended for step-by-step inspection)*

Run the notebooks in order to walk through every stage in detail. Intermediate outputs, plots, and metrics are displayed inline as each cell executes.

```
1. preprocessing.ipynb                 cleans raw CSV, outputs reviews_preprocessed.csv
2. sentiment_analysis.ipynb            trains all 12 configurations, baseline + tuned
3. sentiment_analysis_with_bert.ipynb  BERT-based extension (optional)
```

### 🕷️ Scrape new reviews

```bash
python scrape_playstore_reviews.py
```

<br/>

## 🎓 Acknowledgements

Built as the course project for **CSAI 452 – Natural Language Processing**.  
Dataset collected from the Google Play Store.
