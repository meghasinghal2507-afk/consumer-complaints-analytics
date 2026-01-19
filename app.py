import streamlit as st
import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import pipeline
from keybert import KeyBERT

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Consumer Complaint Analytics", layout="wide")
st.title("📊 Consumer Complaints Analytics Pipeline")

# -----------------------------
# Load Models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    sentiment_model = pipeline("sentiment-analysis")
    keyword_model = KeyBERT()
    return embedder, sentiment_model, keyword_model

embedder, sentiment_model, keyword_model = load_models()

# -----------------------------
# Helper Functions
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_recommendations(df):
    recs = []

    top_cluster = df["cluster"].value_counts().idxmax()
    top_issue = df[df["cluster"] == top_cluster]["topic"].iloc[0]

    neg_ratio = (df["sentiment"] == "NEGATIVE").mean()

    recs.append(f"Most frequent complaint theme: **{top_issue}**.")
    recs.append(f"Negative sentiment ratio is **{round(neg_ratio*100,2)}%**.")

    if neg_ratio > 0.6:
        recs.append("High customer dissatisfaction detected. Immediate process intervention recommended.")

    recs.append("Consider proactive communication and self-service resolution for recurring issues.")

    return recs

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Complaints Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Column Selection
    # -----------------------------
    text_col = st.selectbox("Select Complaint Text Column", df.columns)
    date_col = st.selectbox("Select Date Column", df.columns)

    # -----------------------------
    # Data Cleaning
    # -----------------------------
    df["clean_text"] = df[text_col].apply(clean_text)

    # -----------------------------
    # Embeddings
    # -----------------------------
    with st.spinner("Generating embeddings..."):
        embeddings = embedder.encode(df["clean_text"].tolist())

    # -----------------------------
    # Clustering
    # -----------------------------
    num_clusters = st.slider("Number of Complaint Groups", 3, 10, 5)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)

    # -----------------------------
    # Topic Extraction
    # -----------------------------
    cluster_topics = {}
    for c in df["cluster"].unique():
        texts = df[df["cluster"] == c]["clean_text"].tolist()
        keywords = keyword_model.extract_keywords(
            " ".join(texts), top_n=5
        )
        cluster_topics[c] = ", ".join([k[0] for k in keywords])

    df["topic"] = df["cluster"].map(cluster_topics)

    # -----------------------------
    # Sentiment Analysis
    # -----------------------------
    with st.spinner("Analyzing sentiment..."):
        sentiments = sentiment_model(df["clean_text"].tolist())
        df["sentiment"] = [s["label"] for s in sentiments]

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    st.subheader("📌 Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Complaints", len(df))

    with col2:
        st.metric(
            "Negative Sentiment %",
            f"{round((df['sentiment']=='NEGATIVE').mean()*100,2)}%"
        )

    with col3:
        st.metric("Unique Complaint Themes", df["cluster"].nunique())

    # -----------------------------
    # Complaint Themes
    # -----------------------------
    st.subheader("🧩 Complaint Themes")
    theme_df = (
        df.groupby("topic")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    st.dataframe(theme_df)

    # -----------------------------
    # Sentiment vs Themes
    # -----------------------------
    st.subheader("😡 Sentiment by Theme")
    sentiment_theme = pd.crosstab(df["topic"], df["sentiment"])
    st.dataframe(sentiment_theme)

    # -----------------------------
    # Trends Over Time
    # -----------------------------
    st.subheader("📈 Complaint Trends Over Time")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    trend = df.groupby(df[date_col].dt.to_period("M")).size()
    st.line_chart(trend)

    # -----------------------------
    # Recommendations
    # -----------------------------
    st.subheader("💡 Model Recommendations")
    recommendations = generate_recommendations(df)
    for r in recommendations:
        st.write("- ", r)

    # -----------------------------
    # Download Results
    # -----------------------------
    st.subheader("⬇️ Download Enriched Data")
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="complaint_analysis_results.csv"
    )
