#!/usr/bin/env python3
import streamlit as st
from transformers import pipeline
import torch
import pandas as pd
import altair as alt
from io import BytesIO

# 🚨 Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="✨ Smart Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# ===== Model Config =====
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TASK = "sentiment-analysis"

# ===== Device Selection =====
device = 0 if torch.cuda.is_available() else -1

# ===== Load Model =====
@st.cache_resource
def load_model():
    return pipeline(task=TASK, model=MODEL_ID, device=device)

classifier = load_model()

# ===== Custom CSS =====
st.markdown(
    """
    <style>
    body {
        font-family: "Segoe UI", sans-serif;
        transition: all 0.3s ease-in-out;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #d3d3d3;
        font-size: 16px;
        transition: 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #2a9d8f;
        box-shadow: 0 0 10px rgba(42,157,143,0.3);
    }
    .result-card {
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-top: 15px;
        text-align: center;
        animation: fadeIn 0.7s ease-in-out;
    }
    .sentiment-label {
        font-size: 32px;
        font-weight: 600;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== Header =====
st.title("💬 Smart Sentiment Analyzer")
st.markdown("Analyze **single or multiple sentences** with AI-powered sentiment insights. 🌟")

# ===== Examples =====
samples = [
    "I love this product! It works perfectly. 😍",
    "This is the worst experience ever. 😡",
    "It's okay, not too bad but not great either. 🤔"
]

st.write("### 🔹 Try Quick Examples")
cols = st.columns(len(samples))
for i, sentence in enumerate(samples):
    if cols[i].button(f"💡 {sentence[:20]}..."):
        if "user_input" not in st.session_state or not st.session_state.user_input.strip():
            st.session_state.user_input = sentence
        else:
            st.session_state.user_input += "\n" + sentence

# ===== Text Input (supports multiple lines) =====
user_input = st.text_area(
    "✍️ Enter one or more sentences (one per line):",
    key="user_input",
    height=160,
    placeholder="Type here...\nYou can add multiple sentences, each on a new line."
)

# ===== Prediction =====
if user_input.strip():
    lines = [line.strip() for line in user_input.split("\n") if line.strip()]
    results = classifier(lines, top_k=None if len(lines) > 1 else None)

    if len(lines) == 1:  # Single sentence
        # Normalize results to always be list of dicts
        res = results if isinstance(results, list) else [results]
        res = res[0] if isinstance(res[0], list) else [res[0]]
        df = pd.DataFrame(res)[["label", "score"]]

        # Best result
        best = df.iloc[df["score"].idxmax()]
        label, score = best["label"], best["score"]

        label_map = {
            "Negative": ("❌ Negative", "#ff4d4d"),
            "Neutral": ("⚖️ Neutral", "#f4a261"),
            "Positive": ("✅ Positive", "#2a9d8f")
        }
        label_text, color = label_map.get(label, (label, "#457b9d"))

        st.markdown(
            f"""
            <div class="result-card" style="background: linear-gradient(135deg, {color}20, {color}50);">
                <div class="sentiment-label" style="color:{color};">{label_text}</div>
                <p style="font-size:18px; margin-top:5px;">Confidence: <b>{score:.2%}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show chart only if multiple labels exist
        if len(df) > 1:
            chart_type = st.radio("📊 Choose chart type:", ["Pie Chart", "Bar Chart"], horizontal=True)

            if chart_type == "Pie Chart":
                chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="score", type="quantitative"),
                    color=alt.Color(field="label", type="nominal", scale=alt.Scale(scheme="set1")),
                    tooltip=["label", alt.Tooltip("score", format=".2%")]
                )
            else:
                chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("label", sort="-y"),
                    y=alt.Y("score", axis=alt.Axis(format=".0%")),
                    color=alt.Color("label", scale=alt.Scale(scheme="set1")),
                    tooltip=["label", alt.Tooltip("score", format=".2%")]
                )
            st.altair_chart(chart, use_container_width=True)

    else:  # Batch sentences
        records = []
        for text, res in zip(lines, results):
            best = max(res, key=lambda x: x["score"])
            records.append({
                "Text": text,
                "Sentiment": best["label"],
                "Confidence": f"{best['score']:.2%}"
            })
        df = pd.DataFrame(records)

        st.write("### 📊 Batch Sentiment Results")
        st.dataframe(
            df.style.highlight_max(axis=0, subset=["Confidence"], color="lightgreen"),
            use_container_width=True
        )

        # Download option
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        st.download_button(
            "📥 Download Results as CSV",
            buffer.getvalue(),
            "sentiment_results.csv",
            "text/csv"
        )

        # Avoid duplicate history
        if "history" not in st.session_state:
            st.session_state.history = []
        new_records = [r for r in records if r not in st.session_state.history]
        st.session_state.history.extend(new_records)

    if "history" in st.session_state and st.session_state.history:
        with st.expander("📜 View Analysis History"):
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)

else:
    st.info("💡 Enter text above or try a quick example to begin analysis.")
