import streamlit as st
from textblob import TextBlob
import pandas as pd
import plotly.express as px

# Page setup
st.set_page_config(page_title="Mood Analytics Dashboard", layout="wide")

# ---- Custom UI Styling ----
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: white;
}
.stTextArea textarea {
    background-color: #262730;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Mood Analytics Dashboard")

# ---- Mood Classification Function ----
def classify_mood(p):
    if p > 0.3:
        return "Happy"
    elif p >= -0.3:
        return "Neutral"
    elif p >= -0.6:
        return "Sad"
    else:
        return "Angry/Anxious"

# ---- Sidebar Controls ----
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Mood Shift Threshold", 0.1, 1.0, 0.4)

# ---- Manual Input ----
st.subheader("✍️ Enter Daily Log")
text = st.text_area("Write your thoughts here...")

if st.button("Analyze Mood"):
    if text:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        mood = classify_mood(polarity)

        st.success(f"Detected Mood: {mood}")
        st.info(f"Polarity Score: {polarity:.2f}")
    else:
        st.warning("Please enter some text!")

# ---- File Upload ----
st.subheader("📂 Upload CSV File")
file = st.file_uploader("Upload CSV with column name 'text'", type=["csv"])

if file:
    df = pd.read_csv(file)

    results = []

    for t in df['text']:
        blob = TextBlob(str(t))
        polarity = blob.sentiment.polarity
        mood = classify_mood(polarity)
        results.append({"text": t, "polarity": polarity, "mood": mood})

    result_df = pd.DataFrame(results)

    # ---- Metrics ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Entries", len(result_df))
    col2.metric("Average Polarity", round(result_df['polarity'].mean(), 2))
    col3.metric("Most Common Mood", result_df['mood'].mode()[0])

    # ---- Mood Trend Graph ----
    st.subheader("📈 Mood Trend")
    fig = px.line(result_df, y="polarity", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Mood Shift Detection ----
    shifts = []
    for i in range(1, len(result_df)):
        if abs(result_df['polarity'][i] - result_df['polarity'][i-1]) > threshold:
            shifts.append(i)

    if shifts:
        st.error(f"⚠️ Mood Shifts Detected: {len(shifts)}")
    else:
        st.success("No significant mood shifts detected")

    # ---- Mood Distribution ----
    st.subheader("📊 Mood Distribution")
    fig2 = px.bar(result_df['mood'].value_counts())
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Data Table ----
    st.subheader("📋 Data Table")
    st.dataframe(result_df)

    # ---- Summary ----
    st.subheader("📝 Summary Report")
    st.write(f"""
    Total Logs: {len(result_df)}  
    Average Mood Score: {round(result_df['polarity'].mean(),2)}  
    Most Frequent Mood: {result_df['mood'].mode()[0]}  
    Mood Shifts Detected: {len(shifts)}  
    """)