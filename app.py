import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from transformers import pipeline

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Comcast Complaint Analytics", layout="wide")
st.title("📊 Comcast Consumer Complaint Analytics")

# -----------------------------
# Domain Stopwords
# -----------------------------
DOMAIN_STOPWORDS = [
    "comcast", "xfinity", "customer", "service", "company",
    "call", "phone", "agent", "representative", "account"
]

# -----------------------------
# Load Sentiment Model
# -----------------------------
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_sentiment()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Comcast Complaints Data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    text_col = st.selectbox("Complaint Text Column", df.columns)
    date_col = st.selectbox("Date Column", df.columns)

    # -----------------------------
    # Cleaning
    # -----------------------------
    df["clean_text"] = df[text_col].apply(clean_text)

    # -----------------------------
    # TF-IDF
    # -----------------------------
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=20,
        ngram_range=(1, 2)
    )

    tfidf = vectorizer.fit_transform(df["clean_text"])

    # -----------------------------
    # Topic Modeling (NMF)
    # -----------------------------
    n_topics = st.slider("Number of Complaint Themes", 4, 10, 6)
    nmf = NMF(n_components=n_topics, random_state=42)
    topic_matrix = nmf.fit_transform(tfidf)

    df["topic_id"] = topic_matrix.argmax(axis=1)

    feature_names = vectorizer.get_feature_names_out()

    topic_labels = {}
    for topic_idx, topic in enumerate(nmf.components_):
        keywords = [
            feature_names[i]
            for i in topic.argsort()[:-6:-1]
            if feature_names[i] not in DOMAIN_STOPWORDS
        ]
        topic_labels[topic_idx] = ", ".join(keywords)

    df["topic"] = df["topic_id"].map(topic_labels)

    # -----------------------------
    # Sentiment
    # -----------------------------
    sentiments = sentiment_model(df["clean_text"].tolist())
    df["sentiment"] = [s["label"] for s in sentiments]

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    st.subheader("🔍 Key Complaint Drivers")

    topic_summary = (
        df.groupby("topic")
        .size()
        .reset_index(name="complaints")
        .sort_values("complaints", ascending=False)
    )
    st.dataframe(topic_summary)

    # -----------------------------
    # Sentiment by Topic
    # -----------------------------
    st.subheader("😡 Sentiment by Complaint Type")
    sentiment_table = pd.crosstab(df["topic"], df["sentiment"])
    st.dataframe(sentiment_table)

    # -----------------------------
    # Time Patterns
    # -----------------------------
    st.subheader("📈 Complaint Trends")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    trend = df.groupby(df[date_col].dt.to_period("M")).size()
    st.line_chart(trend)

    # -----------------------------
    # Business Insights
    # -----------------------------
    st.subheader("💡 Actionable Insights")

    top_issue = topic_summary.iloc[0]["topic"]
    neg_ratio = (df["sentiment"] == "NEGATIVE").mean()

    st.write(f"• Highest complaint volume is related to **{top_issue}**.")
    st.write(f"• **{round(neg_ratio*100,2)}%** of complaints express negative sentiment.")

    if "bill" in top_issue:
        st.write("• Billing clarity and post-cancellation charges need immediate review.")
    if "internet" in top_issue:
        st.write("• Frequent service disruptions indicate infrastructure reliability issues.")
    if "support" in top_issue:
        st.write("• Customer support process optimization is recommended.")

    # -----------------------------
    # Download
    # -----------------------------
    st.download_button(
        "Download Enriched Data",
        df.to_csv(index=False),
        "comcast_complaints_enriched.csv"
    )
