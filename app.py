import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Mood Shift Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ HEADER ------------------
st.title("🧠 Mood Shift Detection Dashboard")
st.markdown("Analyze daily emotions using NLP sentiment analysis")

st.divider()

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("📌 Instructions")
    st.write("""
    1. Enter daily logs (one per line)
    2. Click Analyze
    3. View mood trends & shifts
    """)
    st.info("Built using Streamlit + NLTK")

# ------------------ INPUT ------------------
st.subheader("✍️ Enter Daily Logs")

user_input = st.text_area("Write your logs here:")

# ------------------ FUNCTION ------------------
def get_mood(score):
    if score >= 0.05:
        return "😊 Happy"
    elif score <= -0.05:
        return "😔 Sad"
    else:
        return "😐 Neutral"

# ------------------ ANALYZE ------------------
if st.button("🚀 Analyze Mood"):

    if not user_input.strip():
        st.warning("Please enter some logs.")
    else:
        logs = [l.strip() for l in user_input.split("\n") if l.strip()]

        data = []
        scores = []

        for i, log in enumerate(logs):
            score = sia.polarity_scores(log)["compound"]
            mood = get_mood(score)

            data.append([i+1, log, round(score, 3), mood])
            scores.append(score)

        df = pd.DataFrame(data, columns=["Day", "Text", "Score", "Mood"])

        st.divider()

        # ------------------ TOP METRICS ------------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Entries", len(df))
        col2.metric("Most Recent Mood", df["Mood"].iloc[-1])
        col3.metric("Avg Sentiment", round(sum(scores)/len(scores), 2))

        st.divider()

        # ------------------ TABLE ------------------
        st.subheader("📊 Analysis Table")
        st.dataframe(df, use_container_width=True)

        st.divider()

        # ------------------ CHARTS ------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Sentiment Trend")

            fig, ax = plt.subplots()
            ax.plot(range(1, len(scores)+1), scores, marker="o")
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Day")
            ax.set_ylabel("Sentiment Score")
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Mood Distribution")

            mood_counts = df["Mood"].value_counts()
            st.bar_chart(mood_counts)

        st.divider()

        # ------------------ MOOD SHIFT ------------------
        st.subheader("⚡ Mood Shift Detection")

        shifts = []
        for i in range(1, len(scores)):
            if abs(scores[i] - scores[i-1]) > 0.5:
                shifts.append(i+1)

        if shifts:
            st.error(f"Significant mood shifts detected on Day(s): {shifts}")
        else:
            st.success("No major mood shifts detected")

        st.divider()

        # ------------------ DOWNLOAD ------------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Report",
            data=csv,
            file_name="mood_report.csv",
            mime="text/csv"
        )
