import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from nrclex import NRCLex                      # nuanced emotion lexicon
import text2emotion as te                      # coarse five‑emotion set

import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger")
nltk.download("brown")
nltk.download("movie_reviews")
nltk.download("vader_lexicon")

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# ---------------------------- configuration ----------------------------------
CSV_PATH = "Alchemist_Sentences.csv"  # <‑‑ update if needed
SEGMENTS  = 80     # number of equal text segments for Ben‑Schmidt chart
ROLLING_N = 50     # window for smoothing line chart

# ----------------------------- load + clean ----------------------------------
df = pd.read_csv(CSV_PATH)
assert "Sentences" in df.columns, "Expected a 'Sentences' column."

# drop NaNs & ensure string
df = df[df["Sentences"].notna()].copy()
df["Sentences"] = df["Sentences"].astype(str)

# ------------------------ basic polarity visuals -----------------------------
POLAR_COLS = ["Positive", "Negative", "Neutral", "Compound"]

# histogram

def plot_histograms():
    for col in POLAR_COLS:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], bins=30, kde=True, color="#719ba6")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        try:
            import streamlit as st
            st.pyplot(plt)
        except ImportError:
            plt.show()

# categorise sentence‑level polarity
def categorize_sentiment(c):
    return "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral"

df["Sentiment_Category"] = df["Compound"].apply(categorize_sentiment)


def plot_sentence_category_bar():
    import plotly.express as px

    sentiment_counts = df["Sentiment_Category"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        title="Sentence-level Sentiment Category Distribution",
        color_discrete_sequence=["#719ba6"]
    )

    try:
        import streamlit as st
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        fig.show()


# ----------------------- Ben‑Schmidt style chart -----------------------------

def plot_segment_chart(n_segments: int = SEGMENTS):
    df["Segment"] = pd.qcut(df.index, q=n_segments, labels=False)

    grp = df.groupby("Segment")
    seg = pd.DataFrame({
        "Positive": grp["Compound"].apply(lambda x: x[x > 0].sum()),
        "Negative": grp["Compound"].apply(lambda x: x[x < 0].sum()),
        "Mean": grp["Compound"].mean()
    }).fillna(0)

    plt.figure(figsize=(16, 6))
    plt.bar(seg.index, seg["Positive"], color="#94d6a4", label="Positive")
    plt.bar(seg.index, seg["Negative"], color="#d47b7b", label="Negative")
    plt.plot(seg.index, seg["Mean"], color="black", linewidth=2, label="Avg Sentiment")
    plt.title("Sentiment by Book Segment – The Alchemist")
    plt.xlabel("Segment (equal length)")
    plt.ylabel("Sum of Sentiment")
    plt.legend()
    plt.tight_layout()
    try:
        import streamlit as st
        st.pyplot(plt)
    except ImportError:
        plt.show()

# ---------------------------- rolling trend ----------------------------------

def plot_rolling_compound(window: int = ROLLING_N):
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df["Compound"].rolling(window).mean(), color="#719ba6")
    plt.title("Rolling Sentiment Trend Across The Alchemist")
    plt.ylabel("Compound Sentiment (smoothed)")
    plt.xlabel("Sentence Index")
    try:
        import streamlit as st
        st.pyplot(plt)
    except ImportError:
        plt.show()

# --------------- most extreme sentences (for qualitative study) --------------

def print_extremes():
    pos = df.loc[df["Compound"].idxmax()]
    neg = df.loc[df["Compound"].idxmin()]
    print("\nMost Positive Sentence:\n", pos["Sentences"])
    print("\nMost Negative Sentence:\n", neg["Sentences"])

# ---------------------- text2emotion coarse emotions -------------------------

def add_text2emotion():
    try:
        import streamlit as st
        @st.cache_data
        def _cached_text2emotion():
            emo = df["Sentences"].apply(lambda x: te.get_emotion(str(x))).apply(pd.Series)
            df_emo = pd.concat([df, emo], axis=1)
            return df_emo
        return _cached_text2emotion()
    except ImportError:
        emo = df["Sentences"].apply(lambda x: te.get_emotion(str(x))).apply(pd.Series)
        df_emo = pd.concat([df, emo], axis=1)
        return df_emo

# ----------------------- NRC EmoLex nuanced emotions -------------------------

EMO_CATS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust"
]

def add_nrc_emotions():
    try:
        import streamlit as st
        @st.cache_data
        def _cached_nrc_emotions():
            def nrc_freq(sent):
                emo_dict = NRCLex(sent).affect_frequencies
                return {k: emo_dict.get(k, 0.0) for k in EMO_CATS}
            nrc = df["Sentences"].apply(nrc_freq).apply(pd.Series)
            return pd.concat([df, nrc], axis=1)
        return _cached_nrc_emotions()
    except ImportError:
        def nrc_freq(sent):
            emo_dict = NRCLex(sent).affect_frequencies
            return {k: emo_dict.get(k, 0.0) for k in EMO_CATS}
        nrc = df["Sentences"].apply(nrc_freq).apply(pd.Series)
        return pd.concat([df, nrc], axis=1)

# ---------------------- overall NRC distribution plot ------------------------

def plot_nrc_distribution(df_nrc: pd.DataFrame):
    totals = df_nrc[EMO_CATS].sum()
    totals.plot(kind="bar", color="#719ba6")
    plt.title("Overall NRC Emotion Distribution – The Alchemist (English)")
    plt.ylabel("Total Emotion Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    try:
        import streamlit as st
        st.pyplot(plt)
    except ImportError:
        plt.show()

# ------------------------ translation caveat print ---------------------------

def print_translation_note():
    note = (
        "This dashboard is brought to you by Naushin Uddin."
    )
    try:
        import streamlit as st
        st.markdown(note)
    except ImportError:
        print("\n" + note + "\n")
