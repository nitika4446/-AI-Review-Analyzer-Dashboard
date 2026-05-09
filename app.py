# =========================================
# AI REVIEW ANALYZER DASHBOARD
# =========================================

import streamlit as st
import pandas as pd
import plotly.express as px
import re
from collections import Counter

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from transformers import pipeline
from keybert import KeyBERT

# =========================================
# DOWNLOAD NLTK DATA
# =========================================
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Review Analyzer Dashboard",
    page_icon="🤖",
    layout="wide"
)

# =========================================
# TITLE
# =========================================
st.title("🤖 AI Review Analyzer Dashboard")

st.markdown(
    "Analyze Customer Reviews using NLP & Machine Learning"
)

# =========================================
# LOAD MODELS
# =========================================
@st.cache_resource
def load_models():

    # Sentiment Analysis Pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis"
    )

    # Keyword Extraction Model
    kw_model = KeyBERT()

    return sentiment_pipeline, kw_model


# Load Models
sentiment_pipeline, kw_model = load_models()

# =========================================
# CLEAN TEXT FUNCTION
# =========================================
def clean_text(text):

    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z ]', '', text)

    # Tokenize text
    words = word_tokenize(text)

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    # Join cleaned words
    return " ".join(filtered_words)

# =========================================
# FAST ML SENTIMENT
# =========================================
def ml_sentiment(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"

    elif polarity < 0:
        return "Negative"

    else:
        return "Neutral"

# =========================================
# BERT SENTIMENT
# =========================================
def bert_sentiment(text):

    try:

        result = sentiment_pipeline(text[:512])[0]

        return result['label'], round(result['score'], 2)

    except Exception:

        return "UNKNOWN", 0.0

# =========================================
# SIMPLE TEXT SUMMARY
# =========================================
def generate_summary(text):

    text = str(text)

    sentences = text.split('.')

    summary_sentences = sentences[:5]

    summary = '. '.join(summary_sentences)

    return summary

# =========================================
# KEYWORD EXTRACTION
# =========================================
def extract_keywords(text):

    try:

        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=10
        )

        return [kw[0] for kw in keywords]

    except Exception:

        return []

# =========================================
# LOAD DATASET
# =========================================
try:

    df = pd.read_csv("reviews_data.csv")

    st.success("Dataset Loaded Successfully!")

except Exception as e:

    st.error(f"Dataset Loading Error: {e}")

    # Fallback Sample Data
    sample_data = {
        "Review": [
            "Amazing food and excellent service",
            "Very bad delivery experience",
            "Loved the ambience and quality",
            "Terrible customer support",
            "Fast delivery and good packaging",
            "Excellent taste and affordable price",
            "Food quality was average",
            "Highly recommended restaurant",
            "Very disappointing experience",
            "Staff was polite and helpful"
        ]
    }

    df = pd.DataFrame(sample_data)

# =========================================
# DATA PREVIEW
# =========================================
st.subheader("📄 Dataset Preview")

st.dataframe(df.head())

# =========================================
# COLUMN SELECTION
# =========================================
text_column = st.selectbox(
    "Select Review Column",
    df.columns
)

# =========================================
# ANALYZE BUTTON
# =========================================
if st.button("🚀 Analyze Reviews"):

    with st.spinner("Analyzing Reviews..."):

        # Remove Null Values
        df = df.dropna(subset=[text_column])

        # Clean Reviews
        df['Cleaned_Review'] = df[text_column].apply(
            clean_text
        )

        # ML Sentiment
        df['ML_Sentiment'] = df['Cleaned_Review'].apply(
            ml_sentiment
        )

        # BERT Sentiment
        bert_results = df['Cleaned_Review'].apply(
            bert_sentiment
        )

        df['BERT_Sentiment'] = bert_results.apply(
            lambda x: x[0]
        )

        df['Confidence'] = bert_results.apply(
            lambda x: x[1]
        )

        # Combine Text
        full_text = " ".join(
            df['Cleaned_Review'].astype(str).tolist()
        )

        # Generate Summary
        summary = generate_summary(full_text)

        # Extract Keywords
        keywords = extract_keywords(full_text)

    # =========================================
    # SUCCESS MESSAGE
    # =========================================
    st.success("Analysis Completed Successfully!")

    # =========================================
    # METRICS
    # =========================================
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Reviews",
        len(df)
    )

    col2.metric(
        "Positive Reviews",
        (df['BERT_Sentiment'] == 'POSITIVE').sum()
    )

    col3.metric(
        "Negative Reviews",
        (df['BERT_Sentiment'] == 'NEGATIVE').sum()
    )

    # =========================================
    # SENTIMENT DISTRIBUTION
    # =========================================
    st.subheader("📊 Sentiment Distribution")

    sentiment_counts = (
        df['BERT_Sentiment']
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        'Sentiment',
        'Count'
    ]

    fig1 = px.pie(
        sentiment_counts,
        names='Sentiment',
        values='Count',
        title='BERT Sentiment Distribution'
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =========================================
    # CONFIDENCE HISTOGRAM
    # =========================================
    st.subheader("📈 Confidence Score Distribution")

    fig2 = px.histogram(
        df,
        x='Confidence',
        nbins=20,
        title='Confidence Score Histogram'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =========================================
    # SUMMARY
    # =========================================
    st.subheader("📝 AI Generated Summary")

    st.write(summary)

    # =========================================
    # KEYWORDS
    # =========================================
    st.subheader("🔑 Top Keywords")

    keyword_df = pd.DataFrame({
        'Keywords': keywords
    })

    st.dataframe(keyword_df)

    # =========================================
    # KEYWORD CHART
    # =========================================
    keyword_counts = Counter(keywords)

    keyword_chart_df = pd.DataFrame({
        'Keyword': list(keyword_counts.keys()),
        'Count': list(keyword_counts.values())
    })

    fig3 = px.bar(
        keyword_chart_df,
        x='Keyword',
        y='Count',
        title='Keyword Frequency'
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # =========================================
    # DETAILED ANALYSIS
    # =========================================
    st.subheader("📋 Detailed Analysis")

    st.dataframe(
        df[[
            text_column,
            'ML_Sentiment',
            'BERT_Sentiment',
            'Confidence'
        ]]
    )

    # =========================================
    # DOWNLOAD RESULTS
    # =========================================
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇ Download Results",
        data=csv,
        file_name='review_analysis_results.csv',
        mime='text/csv'
    )




