import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# ---------------- SAFE NLP SETUP ----------------
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mood Intelligence Dashboard",
    layout="wide",
    page_icon="🧠"
)

# ---------------- HEADER ----------------
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
        🧠 Mood Intelligence Dashboard
    </h1>
    <p style='text-align: center; color: gray;'>
        AI-powered emotional trend & mood shift analysis
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📊 Control Panel")
    st.write("Analyze your emotional patterns from daily logs.")
    st.info("Built with Streamlit + NLP")

# ---------------- INPUT ----------------
st.subheader("✍️ Enter Daily Logs")

user_input = st.text_area("Write one entry per line")

# ---------------- MOOD FUNCTION ----------------
def get_mood(score):
    if score >= 0.05:
        return "😊 Happy"
    elif score <= -0.05:
        return "😔 Sad"
    else:
        return "😐 Neutral"

# ---------------- ANALYSIS ----------------
if st.button("🚀 Generate Insights"):

    if not user_input.strip():
        st.warning("Please enter logs first.")
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

        st.markdown("## 📌 Key Insights")

        # ---------------- METRIC CARDS ----------------
        col1, col2, col3 = st.columns(3)

        col1.markdown(
            f"<div style='padding:20px; background:#1e1e1e; border-radius:10px;'>"
            f"<h3>Total Entries</h3><h2>{len(df)}</h2></div>",
            unsafe_allow_html=True
        )

        col2.markdown(
            f"<div style='padding:20px; background:#1e1e1e; border-radius:10px;'>"
            f"<h3>Latest Mood</h3><h2>{df['Mood'].iloc[-1]}</h2></div>",
            unsafe_allow_html=True
        )

        col3.markdown(
            f"<div style='padding:20px; background:#1e1e1e; border-radius:10px;'>"
            f"<h3>Avg Score</h3><h2>{round(sum(scores)/len(scores),2)}</h2></div>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        # ---------------- TABLE ----------------
        st.subheader("📊 Detailed Analysis")
        st.dataframe(df, use_container_width=True)

        st.markdown("---")

        # ---------------- CHARTS ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Mood Trend")

            fig, ax = plt.subplots()
            ax.plot(range(1, len(scores)+1), scores, marker="o")
            ax.axhline(0, linestyle="--")
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Mood Distribution")

            mood_counts = df["Mood"].value_counts()
            st.bar_chart(mood_counts)

        st.markdown("---")

        # ---------------- SHIFT DETECTION ----------------
        st.subheader("⚡ Mood Shift Detection")

        shifts = []
        for i in range(1, len(scores)):
            if abs(scores[i] - scores[i-1]) > 0.5:
                shifts.append(i+1)

        if shifts:
            st.error(f"Mood shifts detected on Day(s): {shifts}")
        else:
            st.success("Stable emotional pattern detected")

        # ---------------- DOWNLOAD ----------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Report",
            data=csv,
            file_name="mood_report.csv",
            mime="text/csv"
        )
