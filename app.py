import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import time

# ---------------- SAFE NLP SETUP ----------------
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Emotion Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ---------------- ULTRA UI STYLE ----------------
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.main-title {
    font-size: 48px;
    text-align: center;
    font-weight: 800;
    background: linear-gradient(90deg, #00ffe0, #7a5cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    text-align: center;
    color: #aaa;
    font-size: 18px;
}

.card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}

.stButton>button {
    background: linear-gradient(90deg, #00ffe0, #7a5cff);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🧠 AI Emotion Intelligence System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Ultra-premium NLP Mood Shift Detection Dashboard</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ AI Control Panel")
    st.write("Analyze emotional patterns using advanced NLP.")
    st.success("Powered by Streamlit + VADER NLP")

# ---------------- INPUT ----------------
st.subheader("✍️ Enter Your Emotional Logs")

user_input = st.text_area("Type one entry per line", height=150)

# ---------------- MOOD FUNCTION ----------------
def get_mood(score):
    if score >= 0.05:
        return "😊 Happy"
    elif score <= -0.05:
        return "😔 Sad"
    else:
        return "😐 Neutral"

# ---------------- ANALYSIS ----------------
if st.button("🚀 Run AI Emotion Analysis"):

    if not user_input.strip():
        st.warning("Please enter some text logs.")
    else:

        # ⏳ LOADING ANIMATION
        with st.spinner("AI is analyzing emotions..."):
            time.sleep(1.5)

        logs = [x.strip() for x in user_input.split("\n") if x.strip()]

        data = []
        scores = []

        for i, log in enumerate(logs):
            score = sia.polarity_scores(log)["compound"]
            mood = get_mood(score)

            data.append([i+1, log, round(score, 3), mood])
            scores.append(score)

        df = pd.DataFrame(data, columns=["Day", "Text", "Score", "Mood"])

        st.markdown("## 📊 AI Dashboard Insights")

        # ---------------- METRIC CARDS ----------------
        col1, col2, col3 = st.columns(3)

        col1.markdown(f"<div class='card'><h3>📌 Total Logs</h3><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'><h3>🧠 Latest Mood</h3><h2>{df['Mood'].iloc[-1]}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'><h3>📊 Avg Score</h3><h2>{round(sum(scores)/len(scores),2)}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")

        # ---------------- TABLE ----------------
        st.subheader("📋 Emotion Analysis Table")
        st.dataframe(df, use_container_width=True)

        st.markdown("---")

        # ---------------- CHARTS ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Emotion Trend AI Graph")
            fig, ax = plt.subplots()
            ax.plot(range(1, len(scores)+1), scores, marker="o", linewidth=2)
            ax.axhline(0, linestyle="--", color="gray")
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Mood Distribution")
            st.bar_chart(df["Mood"].value_counts())

        st.markdown("---")

        # ---------------- SHIFT DETECTION ----------------
        st.subheader("⚡ AI Mood Shift Engine")

        shifts = []
        for i in range(1, len(scores)):
            if abs(scores[i] - scores[i-1]) > 0.5:
                shifts.append(i+1)

        if shifts:
            st.error(f"⚠ Significant emotional shifts detected on Day(s): {shifts}")
        else:
            st.success("Emotional pattern is stable ✅")

        st.markdown("---")

        # ---------------- DOWNLOAD ----------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download AI Report",
            data=csv,
            file_name="ultra_ai_mood_report.csv",
            mime="text/csv"
        )
