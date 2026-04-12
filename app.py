"""
Sentiment Analysis Studio — Streamlit GUI

Features
    1. Load reviews_dataset.csv
    2. Train feature/classifier combinations and display metrics
    3. Enter a new review and predict its sentiment

Usage:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm          import LinearSVC
from sklearn.naive_bayes  import MultinomialNB, GaussianNB

from preprocessing       import preprocess
from sentiment_analysis  import (
    prepare_data, build_tfidf, build_word2vec,
    evaluate_model, tune_model, get_top_features, tokens_to_vector,
)


# ═════════════════════════════════════════════════════════════════════
# Page configuration
# ═════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sentiment Analysis Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS -------------------------------------------------------
st.markdown("""
<style>
/* Remove top-right Streamlit menu/toolbar (theme switcher lives there) */
[data-testid="stToolbar"] {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

/* ══════════════════════════════════════════════
   LIGHT MODE tokens  (prefers-color-scheme: light
   OR Streamlit's [theme] base = "light")
   ══════════════════════════════════════════════ */
:root {
    --bg-base:        #f8fbff;
    --bg-surface:     rgba(255,255,255,0.86);
    --bg-raised:      rgba(255,255,255,0.95);
    --bg-hover:       rgba(37,99,235,0.10);

    --border-subtle:  rgba(37,99,235,0.14);
    --border-default: rgba(37,99,235,0.25);
    --border-accent:  rgba(37,99,235,0.30);

    --text-primary:   #111827;
    --text-secondary: #475569;
    --text-muted:     #94a3b8;

    --accent-start:   #667eea;
    --accent-end:     #764ba2;

    --badge-pos:      #15803d;
    --badge-neg:      #b91c1c;
    --badge-neu:      #a16207;

    --metric-bg-a:    rgba(59,130,246,0.12);
    --metric-bg-b:    rgba(14,165,233,0.08);
    --metric-border:  rgba(37,99,235,0.20);

    --sidebar-bg:     rgba(255,255,255,0.86);
    --tab-list-bg:    rgba(255,255,255,0.66);

    --result-card-a:  rgba(59,130,246,0.16);
    --result-card-b:  rgba(14,165,233,0.10);
    --h1-start: #c8b6ff; --h1-mid: #b8c0ff; --h1-end: #7aa7ff;
    --result-grad-start: #c8b6ff; --result-grad-end: #7aa7ff;
}

/* ══════════════════════════════════════════════
   DARK MODE tokens
   ══════════════════════════════════════════════ */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-base:        #0f1117;
        --bg-surface:     #1a1d27;
        --bg-raised:      #20243a;
        --bg-hover:       #252942;

        --border-subtle:  rgba(139,158,255,0.12);
        --border-default: rgba(139,158,255,0.22);
        --border-accent:  rgba(139,158,255,0.40);

        --text-primary:   #e8eaf6;
        --text-secondary: #9399b8;
        --text-muted:     #5d6380;

        --accent-start:   #7c6ff7;
        --accent-end:     #5ea5f7;

        --badge-pos:      #4ade80;
        --badge-neg:      #f87171;
        --badge-neu:      #fbbf24;

        --metric-bg-a:    rgba(124,111,247,0.10);
        --metric-bg-b:    rgba(94,165,247,0.06);
        --metric-border:  rgba(124,111,247,0.25);

        --sidebar-bg:     #1a1d27;
        --tab-list-bg:    #1a1d27;

        --result-card-a:  rgba(124,111,247,0.15);
        --result-card-b:  rgba(94,165,247,0.10);
        --h1-start: #a78bfa; --h1-mid: #818cf8; --h1-end: #60a5fa;
        --result-grad-start: #a78bfa; --result-grad-end: #60a5fa;
    }
}

/* ══════════════════════════════════════════════
   Shared structural styles (theme-agnostic)
   ══════════════════════════════════════════════ */

/* Page background */
.stApp {
    background: radial-gradient(ellipse at top left,
        var(--bg-base) 0%, var(--bg-base) 48%, var(--bg-base) 100%);
}

/* Global text */
.stApp, .stApp p, .stApp label, .stApp li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
.stCaption, [data-testid="stCaptionContainer"] p {
    color: var(--text-primary) !important;
}

/* Headings */
h1 {
    background: linear-gradient(90deg,
        var(--h1-start) 0%, var(--h1-mid) 50%, var(--h1-end) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800 !important;
    font-size: 2.8rem !important;
    letter-spacing: -1px;
    margin-bottom: 0.25rem !important;
}
h2, h3, h4 { color: var(--text-primary) !important; font-weight: 700 !important; }

.subtitle {
    color: var(--text-secondary) !important;
    font-size: 0.95rem;
    margin-top: -0.4rem;
    margin-bottom: 1.2rem;
    letter-spacing: 0.2px;
}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-subtle) !important;
    backdrop-filter: blur(8px);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    margin-top: 1.2rem !important;
}
[data-testid="stSidebar"] .stMarkdown h4 {
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-top: 1rem !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg,
        var(--metric-bg-a) 0%, var(--metric-bg-b) 100%) !important;
    border: 1px solid var(--metric-border) !important;
    padding: 18px 20px !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 24px rgba(10,15,30,0.10) !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div {
    color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, var(--accent-start) 0%, var(--accent-end) 100%) !important;
    color: white !important;
    border: none !important;
    padding: 10px 28px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(102,126,234,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(102,126,234,0.45) !important;
    filter: brightness(1.1) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px !important;
    background: var(--tab-list-bg) !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 10px 22px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--bg-hover) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, var(--accent-start), var(--accent-end)) !important;
    color: white !important;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stSelectbox > div > div,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
}
.stTextArea textarea { font-family: 'Segoe UI', sans-serif !important; }

/* Dropdown menus */
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="select"] ul {
    background-color: var(--bg-raised) !important;
    border: 1px solid var(--border-default) !important;
}
[data-baseweb="menu"] li,
[data-baseweb="option"] {
    background-color: var(--bg-raised) !important;
    color: var(--text-primary) !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="option"]:hover {
    background-color: var(--bg-hover) !important;
}

/* Status widget */
[data-testid="stStatusWidget"],
[data-testid="stStatus"] {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-subtle) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
}

/* Code blocks */
.stCode, code, pre {
    background-color: var(--bg-raised) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
}

/* Dividers */
hr { border-color: var(--border-subtle) !important; }

/* Prediction result card */
.result-card {
    background: linear-gradient(135deg,
        var(--result-card-a) 0%, var(--result-card-b) 100%);
    border: 1px solid var(--border-accent);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 8px 28px rgba(10,15,30,0.12);
}
.result-label {
    font-size: 0.85rem;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
.result-value {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg,
        var(--result-grad-start) 0%, var(--result-grad-end) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.badge-pos {
    -webkit-text-fill-color: var(--badge-pos) !important;
    color: var(--badge-pos) !important;
}
.badge-neg {
    -webkit-text-fill-color: var(--badge-neg) !important;
    color: var(--badge-neg) !important;
}
.badge-neu {
    -webkit-text-fill-color: var(--badge-neu) !important;
    color: var(--badge-neu) !important;
}

.model-stats {
    margin-top: 6px;
    color: var(--text-secondary) !important;
    font-size: 0.85rem;
}

.app-footer {
    text-align: center;
    color: var(--text-muted) !important;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# Matplotlib chart theme — follows active frontend-resolved theme
# ═════════════════════════════════════════════════════════════════════
def _is_dark():
    # Prefer the active frontend theme reported by Streamlit.
    try:
        base = (st.context.theme.base or "").strip().lower()
        if base in {"dark", "light"}:
            return base == "dark"
    except Exception:
        pass

    # Last-resort heuristic using background luminance.
    try:
        bg = st.get_option("theme.backgroundColor") or ""
        if bg.startswith("#") and len(bg) >= 7:
            return int(bg[1:3], 16) < 128
    except Exception:
        pass
    return False

def apply_mpl_theme():
    if _is_dark():
        plt.rcParams.update({
            "figure.facecolor":  "#1a1d27",
            "axes.facecolor":    "#20243a",
            "axes.edgecolor":    "#3a3f5c",
            "axes.labelcolor":   "#9399b8",
            "axes.titlecolor":   "#e8eaf6",
            "xtick.color":       "#9399b8",
            "ytick.color":       "#9399b8",
            "text.color":        "#e8eaf6",
            "grid.color":        "#2d3150",
            "grid.alpha":        0.6,
            "savefig.facecolor": "#1a1d27",
        })
    else:
        plt.rcParams.update({
            "figure.facecolor":  "#ffffff",
            "axes.facecolor":    "#f8fafc",
            "axes.edgecolor":    "#cbd5e1",
            "axes.labelcolor":   "#1f2937",
            "axes.titlecolor":   "#111827",
            "xtick.color":       "#374151",
            "ytick.color":       "#374151",
            "text.color":        "#111827",
            "grid.color":        "#e2e8f0",
            "grid.alpha":        1.0,
            "savefig.facecolor": "#ffffff",
        })

apply_mpl_theme()


# ═════════════════════════════════════════════════════════════════════
# Session state defaults
# ═════════════════════════════════════════════════════════════════════
_defaults = {
    "df_raw":           None,
    "df_preprocessed":  None,
    "data_bundle":      None,
    "features":         {},
    "models":           {},
    "last_csv_path":    "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════
def run_preprocessing(df, progress_callback=None):
    """Apply the preprocessing pipeline to every review in `df`."""
    results = []
    n = len(df)
    for i, text in enumerate(df["review_text"].values):
        results.append(preprocess(text))
        if progress_callback and (i % 200 == 0 or i == n - 1):
            progress_callback((i + 1) / n)

    results_df = pd.DataFrame(results)
    out = df.copy()
    out["cleaned_text"] = results_df["cleaned_text"]
    out["tokens"]       = results_df["tokens"]
    out["token_count"]  = results_df["token_count"]
    return out


def style_confusion(cm, labels, ax, title):
    import matplotlib.patheffects as path_effects

    apply_mpl_theme()
    lc   = "#090a0d" if _is_dark() else "#E9F0F7"
    ann  = "#f1f1f1"  # keep text light; outline will handle contrast
    muted = "#9399b8" if _is_dark() else "#475569"

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="mako",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=False, linewidths=0.6, linecolor=lc,
        annot_kws={"fontsize": 10, "fontweight": "bold", "color": ann},
    )

    # 🔥 Apply outline to all annotation texts
    for text in ax.texts:
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()
        ])

    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted", fontsize=9, color=muted)
    ax.set_ylabel("Actual",    fontsize=9, color=muted)
    ax.tick_params(axis="x", rotation=30, labelsize=8, colors=muted)
    ax.tick_params(axis="y", rotation=0,  labelsize=8, colors=muted)


def label_badge(label):
    l = label.lower()
    if "pos" in l:
        return "badge-pos"
    if "neg" in l:
        return "badge-neg"
    return "badge-neu"


# ═════════════════════════════════════════════════════════════════════
# Header
# ═════════════════════════════════════════════════════════════════════
st.markdown("# Sentiment Analysis Studio")
st.markdown(
    "<div class='subtitle'>"
    "Load &nbsp;•&nbsp; Preprocess &nbsp;•&nbsp; Train &nbsp;•&nbsp; Predict "
    "— all powered by your two notebooks."
    "</div>",
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════
# Sidebar — configuration
# ═════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Dataset")
    csv_path = st.text_input("CSV path", value="reviews_dataset.csv",
                             label_visibility="collapsed")
    sample_size = st.slider(
        "Sample size (0 = all rows)", 0, 30_000, 0, step=1_000,
        help="Use a subset for quick experimentation. 0 means the full dataset.",
    )

    st.markdown("### Features")
    use_tfidf = st.checkbox("TF-IDF  (1-2 grams)", value=True)
    use_w2v   = st.checkbox("Word2Vec  (averaged)", value=True)

    st.markdown("### Classifiers")
    use_lr  = st.checkbox("Logistic Regression", value=True)
    use_svm = st.checkbox("Linear SVM",          value=True)
    use_nb  = st.checkbox("Naive Bayes",         value=True)

    st.markdown("### Training")
    do_tuning = st.checkbox("Hyperparameter tuning (GridSearchCV)", value=False)
    test_size = st.slider("Test split", 0.10, 0.40, 0.20, 0.05)

    st.markdown("---")
    st.caption(
        "Tip: turn tuning on for the final run. Baselines train in seconds, "
        "tuned models take longer."
    )


# ═════════════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════════════
tab_data, tab_train, tab_predict = st.tabs([
    "📁  Dataset",
    "🧪  Train models",
    "🎯  Predict sentiment",
])


# ─── Tab 1: Dataset ───────────────────────────────────────────────────
with tab_data:
    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.markdown("### Load &amp; preprocess")
        st.caption(
            "Reads the CSV from disk, then runs the full cleaning pipeline "
            "(lowercase → URLs → HTML → emojis → punctuation → tokenise → "
            "stopwords → lemmatise)."
        )
    with top_r:
        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("Load dataset", width="stretch")

    if load_btn:
        if not os.path.exists(csv_path):
            st.error(f"CSV file not found: `{csv_path}`")
        else:
            try:
                with st.spinner("Reading CSV..."):
                    df = pd.read_csv(csv_path)
                if sample_size and sample_size < len(df):
                    df = df.sample(n=sample_size, random_state=42) \
                           .reset_index(drop=True)
                st.session_state.df_raw        = df
                st.session_state.last_csv_path = csv_path
                # reset downstream artefacts
                st.session_state.df_preprocessed = None
                st.session_state.data_bundle    = None
                st.session_state.features       = {}
                st.session_state.models         = {}
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

            if st.session_state.df_raw is not None:
                df = st.session_state.df_raw
                st.info(
                    f"Preprocessing **{len(df):,}** reviews — "
                    "please wait, this runs NLTK lemmatisation."
                )
                prog = st.progress(0.0)
                status_txt = st.empty()

                def _cb(frac):
                    prog.progress(min(1.0, frac))
                    status_txt.caption(f"  {int(frac*100)} %  —  cleaning reviews…")

                df_pp = run_preprocessing(df, _cb)
                st.session_state.df_preprocessed = df_pp
                prog.progress(1.0)
                status_txt.empty()
                st.success(
                    f"Preprocessed **{len(df_pp):,}** reviews. "
                    "Head over to the **Train models** tab."
                )

    # Show dataset stats if loaded
    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total reviews", f"{len(df):,}")
        m2.metric("Columns",       len(df.columns))
        m3.metric("Classes",       df["label"].nunique() if "label" in df else "—")
        m4.metric(
            "Sources",
            df["source"].nunique() if "source" in df.columns else "—",
        )

        st.markdown("#### Preview")
        preview_df = (
            st.session_state.df_preprocessed
            if st.session_state.df_preprocessed is not None
            else df
        )
        show_cols = [
            c for c in
            ["review_id", "source", "product_category", "review_text",
             "cleaned_text", "rating", "label"]
            if c in preview_df.columns
        ]
        st.dataframe(
            preview_df[show_cols].head(10),
            width="stretch",
            height=320,
        )

        if "label" in df.columns:
            st.markdown("#### Distributions")
            c_a, c_b = st.columns(2)

            with c_a:
                counts = df["label"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 3.3))
                colors = sns.color_palette("mako_r", len(counts))
                bars = ax.barh(counts.index, counts.values, color=colors)
                ax.set_title("Label distribution", fontweight="bold", pad=12)
                ax.set_xlabel("Count")
                for b, v in zip(bars, counts.values):
                    ax.text(
                        v, b.get_y() + b.get_height() / 2,
                        f" {v:,}", va="center", fontsize=9,
                        color="#e8eaf6",
                    )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, width="stretch")

            with c_b:
                if "rating" in df.columns:
                    rc = df["rating"].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(6, 3.3))
                    ax.bar(
                        rc.index.astype(str), rc.values,
                        color=sns.color_palette("viridis", len(rc)),
                    )
                    ax.set_title("Rating distribution", fontweight="bold", pad=12)
                    ax.set_xlabel("Rating")
                    ax.set_ylabel("Count")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, width="stretch")
    else:
        st.info("Click **Load dataset** above to read the CSV and run preprocessing.")


# ─── Tab 2: Train models ─────────────────────────────────────────────
with tab_train:
    if st.session_state.df_preprocessed is None:
        st.warning("Load and preprocess the dataset first (see the **Dataset** tab).")
    else:
        st.markdown("### Train &amp; evaluate")
        st.caption(
            "Train the feature × classifier combinations chosen in the sidebar. "
            "Each run produces accuracy, precision, recall, F1-macro and a "
            "confusion matrix."
        )

        if not (use_tfidf or use_w2v):
            st.error("Select at least one feature representation in the sidebar.")
        elif not (use_lr or use_svm or use_nb):
            st.error("Select at least one classifier in the sidebar.")
        else:
            train_btn = st.button("Start training", type="primary")

            if train_btn:
                df_pp = st.session_state.df_preprocessed

                with st.status("Running training pipeline…", expanded=True) as status:
                    # 1) split
                    st.write("• Preparing data (stratified split)…")
                    bundle = prepare_data(df_pp, test_size=test_size)
                    st.session_state.data_bundle = bundle
                    st.write(
                        f"  train = {len(bundle['X_train_text']):,}   "
                        f"test = {len(bundle['X_test_text']):,}   "
                        f"classes = {bundle['labels']}"
                    )

                    # 2) build features
                    features = {}
                    if use_tfidf:
                        st.write("• Building TF-IDF features…")
                        tfidf, X_tr, X_te = build_tfidf(
                            bundle["X_train_text"], bundle["X_test_text"]
                        )
                        features["TF-IDF"] = {
                            "vec": tfidf, "X_train": X_tr, "X_test": X_te,
                        }
                        st.write(f"  vocab = {len(tfidf.vocabulary_):,}")

                    if use_w2v:
                        st.write("• Training Word2Vec…")
                        w2v, X_tr_w, X_te_w = build_word2vec(
                            bundle["df"], bundle["train_pos"], bundle["test_pos"]
                        )
                        features["Word2Vec"] = {
                            "model": w2v, "X_train": X_tr_w, "X_test": X_te_w,
                            "dim": w2v.vector_size,
                        }
                        st.write(
                            f"  vocab = {len(w2v.wv):,}   dim = {w2v.vector_size}"
                        )

                    st.session_state.features = features

                    # 3) model matrix
                    y_train = bundle["y_train"]
                    y_test  = bundle["y_test"]
                    labels  = bundle["labels"]

                    configs = []
                    if use_lr:  configs.append(("lr",  "Logistic Regression"))
                    if use_svm: configs.append(("svm", "Linear SVM"))
                    if use_nb:  configs.append(("nb",  "Naive Bayes"))

                    results = {}
                    total = len(configs) * len(features)
                    step  = 0
                    bar   = st.progress(0.0)

                    for feat_name, feat_data in features.items():
                        X_tr = feat_data["X_train"]
                        X_te = feat_data["X_test"]

                        for cid, cname in configs:
                            step += 1
                            full_name = f"{cname} + {feat_name}"
                            st.write(f"• Training {full_name} ({step}/{total})…")

                            if cid == "lr":
                                factory = lambda: LogisticRegression(
                                    max_iter=1000, random_state=42)
                                pgrid = {"C": [0.1, 1, 10]}

                            elif cid == "svm":
                                factory = lambda: LinearSVC(
                                    C=1.0, max_iter=3000, random_state=42)
                                pgrid = {"C": [0.1, 1, 10]}

                            else:  # nb
                                if feat_name == "TF-IDF":
                                    factory = lambda: MultinomialNB(alpha=0.5)
                                    pgrid   = {"alpha": [0.1, 0.5, 1.0, 5.0]}
                                else:
                                    factory = lambda: GaussianNB()
                                    pgrid   = {"var_smoothing": [1e-9, 1e-8, 1e-7]}

                            if do_tuning:
                                r = tune_model(
                                    factory, pgrid, X_tr, X_te,
                                    y_train, y_test, labels, full_name,
                                )
                            else:
                                r = evaluate_model(
                                    factory(), X_tr, X_te,
                                    y_train, y_test, labels, full_name,
                                )

                            results[full_name] = {
                                "result":  r,
                                "feature": feat_name,
                            }
                            bar.progress(step / total)

                    st.session_state.models = results
                    status.update(
                        label=f"Training complete — {len(results)} models",
                        state="complete", expanded=False,
                    )

                st.success(
                    f"Trained **{len(st.session_state.models)}** model configurations."
                )

            # ── Display results if we have any ───────────────────────
            if st.session_state.models:
                models = st.session_state.models
                labels = st.session_state.data_bundle["labels"]

                st.markdown("---")
                st.markdown("### Performance summary")

                rows = []
                for name, mdata in models.items():
                    r = mdata["result"]
                    rows.append({
                        "Model":     name,
                        "Accuracy":  r["acc"],
                        "Precision": r["prec"],
                        "Recall":    r["rec"],
                        "F1-macro":  r["f1_macro"],
                    })
                results_df = pd.DataFrame(rows).sort_values(
                    "F1-macro", ascending=False,
                ).reset_index(drop=True)

                st.dataframe(
                    results_df.style
                        .format({
                            "Accuracy":  "{:.4f}",
                            "Precision": "{:.4f}",
                            "Recall":    "{:.4f}",
                            "F1-macro":  "{:.4f}",
                        })
                        .background_gradient(
                            cmap="mako_r",
                            subset=["Accuracy", "Precision", "Recall", "F1-macro"],
                        ),
                    width="stretch",
                    hide_index=True,
                )

                # Best model headline
                best_name = results_df.iloc[0]["Model"]
                best_r    = models[best_name]["result"]

                st.markdown(f"#### Best model — `{best_name}`")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",  f"{best_r['acc']:.4f}")
                c2.metric("Precision", f"{best_r['prec']:.4f}")
                c3.metric("Recall",    f"{best_r['rec']:.4f}")
                c4.metric("F1-macro",  f"{best_r['f1_macro']:.4f}")

                if do_tuning and "best_params" in best_r:
                    st.caption(f"Best params: `{best_r['best_params']}`")

                # Confusion matrices grid
                st.markdown("#### Confusion matrices")
                n        = len(models)
                n_cols   = min(3, n)
                n_rows   = (n + n_cols - 1) // n_cols
                fig      = plt.figure(figsize=(5.2 * n_cols, 4.0 * n_rows))
                fig.patch.set_facecolor("#2C3244")
                for idx, (name, mdata) in enumerate(models.items()):
                    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                    style_confusion(mdata["result"]["cm"], labels, ax, name)
                plt.tight_layout()
                st.pyplot(fig, width="stretch")

                # Per-model details
                with st.expander("Per-model classification reports"):
                    for name, mdata in models.items():
                        r = mdata["result"]
                        st.markdown(f"**{name}**")
                        st.code(r["report_text"])

                # Top features (linear + TF-IDF only)
                tfidf_linear = [
                    (n, m) for n, m in models.items()
                    if m["feature"] == "TF-IDF"
                    and hasattr(m["result"]["clf"], "coef_")
                ]
                if tfidf_linear:
                    with st.expander("Top predictive features (linear models + TF-IDF)"):
                        for name, mdata in tfidf_linear:
                            st.markdown(f"**{name}**")
                            tfidf_vec = st.session_state.features["TF-IDF"]["vec"]
                            top = get_top_features(
                                mdata["result"]["clf"], tfidf_vec, labels, top_n=12,
                            )
                            if top:
                                for lab, d in top.items():
                                    st.markdown(f"• **{lab}**")
                                    st.markdown(
                                        f"&nbsp;&nbsp;&nbsp;&nbsp;"
                                        f"↑ {', '.join(d['positive'][:12])}"
                                    )
                                    st.markdown(
                                        f"&nbsp;&nbsp;&nbsp;&nbsp;"
                                        f"↓ {', '.join(d['negative'][:12])}"
                                    )


# ─── Tab 3: Predict sentiment ─────────────────────────────────────────
with tab_predict:
    if not st.session_state.models:
        st.warning("Train at least one model first (see the **Train models** tab).")
    else:
        models       = st.session_state.models
        data_bundle  = st.session_state.data_bundle
        le           = data_bundle["le"]
        labels       = data_bundle["labels"]

        st.markdown("### Predict sentiment for a new review")
        st.caption(
            "Type or paste a review. It will be pushed through the same "
            "preprocessing pipeline as the training data, then classified "
            "by the model you choose below."
        )

        col_left, col_right = st.columns([3, 2])
        with col_left:
            review_text = st.text_area(
                "Review text",
                placeholder=(
                    "e.g.  The graphics are stunning and gameplay is smooth, "
                    "but the recent update introduced a game-breaking bug…"
                ),
                height=180,
            )

        with col_right:
            # rank models by F1 so the best is first
            ranked = sorted(
                models.items(),
                key=lambda kv: kv[1]["result"]["f1_macro"],
                reverse=True,
            )
            model_names = [n for n, _ in ranked]
            selected = st.selectbox("Model", model_names, index=0)

            selected_meta = models[selected]
            selected_r    = selected_meta["result"]
            st.markdown(
                f"<div class='model-stats'>"
                f"F1-macro: <b>{selected_r['f1_macro']:.4f}</b> &nbsp;|&nbsp; "
                f"Accuracy: <b>{selected_r['acc']:.4f}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("Predict", type="primary", width="stretch")

        if predict_btn:
            if not review_text.strip():
                st.warning("Please enter some review text.")
            else:
                model        = selected_r["clf"]
                feature_type = selected_meta["feature"]

                # 1) preprocess the input
                pp       = preprocess(review_text)
                cleaned  = pp["cleaned_text"]
                tokens   = pp["tokens"]

                if not tokens:
                    st.warning(
                        "The review has no usable tokens after preprocessing. "
                        "Try a longer or more descriptive sentence."
                    )
                else:
                    # 2) build the feature vector
                    if feature_type == "TF-IDF":
                        vec = st.session_state.features["TF-IDF"]["vec"]
                        x   = vec.transform([cleaned])
                    else:  # Word2Vec
                        w2v = st.session_state.features["Word2Vec"]["model"]
                        dim = st.session_state.features["Word2Vec"]["dim"]
                        x   = tokens_to_vector(tokens, w2v, dim=dim).reshape(1, -1)

                    # 3) predict
                    pred_idx = model.predict(x)[0]
                    pred_lbl = le.inverse_transform([pred_idx])[0]

                    badge = label_badge(pred_lbl)
                    st.markdown(
                        f"""
                        <div class='result-card'>
                            <div class='result-label'>Predicted sentiment</div>
                            <div class='result-value {badge}'>{pred_lbl.upper()}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 4) probability / confidence visualisation
                    probs = None
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(x)[0]
                    elif hasattr(model, "decision_function"):
                        raw = model.decision_function(x)
                        if raw.ndim == 1:
                            exp = np.exp([-raw[0], raw[0]])
                        else:
                            exp = np.exp(raw[0] - raw[0].max())
                        probs = exp / exp.sum()

                    if probs is not None:
                        st.markdown("#### Class confidence")
                        prob_df = pd.DataFrame({
                            "Class":      le.inverse_transform(np.arange(len(probs))),
                            "Confidence": probs,
                        }).sort_values("Confidence", ascending=False)

                        fig, ax = plt.subplots(figsize=(8, 2.2 + 0.4 * len(probs)))
                        colors = sns.color_palette("mako_r", len(prob_df))
                        bars = ax.barh(
                            prob_df["Class"], prob_df["Confidence"], color=colors,
                        )
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Confidence")
                        ax.invert_yaxis()
                        for b, v in zip(bars, prob_df["Confidence"].values):
                            ax.text(
                                v + 0.01, b.get_y() + b.get_height() / 2,
                                f"{v:.1%}", va="center", fontsize=10,
                                color="#0c0d0d",
                                fontweight="bold",
                            )
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig, width="stretch")

                    with st.expander("Preprocessing details"):
                        st.markdown(
                            f"**Cleaned text:**  \n`{cleaned or '(empty)'}`"
                        )
                        st.markdown(
                            f"**Tokens ({pp['token_count']}):**  \n"
                            f"{' · '.join(tokens) if tokens else '(none)'}"
                        )


# ═════════════════════════════════════════════════════════════════════
# Footer
# ═════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div class='app-footer'>"
    "Sentiment Analysis Studio &nbsp;·&nbsp; "
    "Streamlit GUI built on top of <code>preprocessing.py</code> + "
    "<code>sentiment_analysis.py</code>"
    "</div>",
    unsafe_allow_html=True,
)
