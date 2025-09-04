#!/usr/bin/env python3
import os
import re
import json
import time
import numpy as np
import pandas as pd
import torch
import altair as alt
import streamlit as st
from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ==========================
# Page & Theme
# ==========================
st.set_page_config(
    page_title="✨ Smart Sentiment Analyzer — Pro",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Subtle CSS polish
st.markdown("""
<style>
/* App-wide polish */
section.main > div { padding-top: 1rem; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.badge {
  display:inline-flex; align-items:center; gap:.5rem;
  padding:.35rem .6rem; border-radius:999px; font-weight:600; font-size:.9rem;
}
.badge.green { background:#e6f7ec; color:#0c6; }
.badge.red   { background:#fdeaea; color:#d00; }
.badge.orange{ background:#fff3e6; color:#f60; }
.metric-card {
  border-radius:16px; padding:1rem 1.2rem; background:var(--background-color,#fff);
  border:1px solid rgba(0,0,0,.06); box-shadow:0 1px 3px rgba(0,0,0,.05);
}
hr.soft { border:none; border-top:1px solid rgba(0,0,0,.08); margin:1rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ==========================
# Constants & Model Registry
# ==========================
DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_CHOICES = {
    "CardiffNLP — Twitter RoBERTa (EN)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "DistilBERT SST-2 (EN)": "distilbert-base-uncased-finetuned-sst-2-english",
    "BERT-tiny SST-2 (very small, EN)": "prajjwal1/bert-tiny"
}
TASK = "sentiment-analysis"
DEVICE = 0 if torch.cuda.is_available() else -1

# Pretty maps for labels (case-insensitive)
LABEL_PRETTY = {
    "negative": ("❌ Negative", "red"),
    "neutral":  ("⚖️ Neutral",  "orange"),
    "positive": ("✅ Positive", "green"),
}
ORDER = ["negative", "neutral", "positive"]  # ordering for charts

# ==========================
# Cache model loader
# ==========================
@st.cache_resource(show_spinner="Downloading & loading model…")
def load_model(model_id: str):
    """
    Cache the pipeline to avoid reloading on every rerun.
    We don't set return_all_scores here; we pass it at inference.
    """
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    clf = pipeline(task=TASK, model=mdl, tokenizer=tok, device=DEVICE)
    return clf

# ==========================
# Small helpers
# ==========================
def normalize_label(lbl: str) -> str:
    """Map model labels like 'LABEL_0' or 'Negative' to lowercase canonical strings."""
    l = lbl.strip().lower()
    # Some models use 'label_0/1/2' -> we attempt to map by known order if provided
    if re.match(r"^label_\d+$", l):
        # Heuristic: many SST-2 models use 0=NEG, 1=POS
        # We'll map label_0 -> negative, label_1 -> positive, label_2 -> neutral (fallback)
        idx = int(l.split("_")[-1])
        return ["negative", "positive", "neutral"][idx] if idx in [0,1,2] else "neutral"
    # Else pass through (e.g., 'positive', 'neutral', 'negative')
    return l

def pick_top_label(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return best label dict from a list of {'label','score'}."""
    return max(scores, key=lambda d: d["score"])

def scores_to_df(scores: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of {'label','score'} to tidy DataFrame with canonical labels."""
    rows = []
    for d in scores:
        rows.append({
            "label_raw": d["label"],
            "label": normalize_label(d["label"]),
            "score": float(d["score"])
        })
    df = pd.DataFrame(rows)
    # Keep only known labels; if others appear, keep them but after known order
    cat = [x for x in ORDER if x in df["label"].tolist()]
    rest = [x for x in df["label"].unique().tolist() if x not in cat]
    df["label"] = pd.Categorical(df["label"], categories=cat + rest, ordered=True)
    df = df.sort_values("label")
    return df

def styled_badge(label_lower: str) -> str:
    txt, color = LABEL_PRETTY.get(label_lower, (label_lower.title(), "orange"))
    return f"<span class='badge {color}'>{txt}</span>"

def altair_donut(df: pd.DataFrame, title: str = "") -> alt.Chart:
    df = df.copy()
    df["pct"] = df["score"] / df["score"].sum()
    base = alt.Chart(df).encode(
        theta=alt.Theta("pct:Q"),
        color=alt.Color("label:N"),
        tooltip=[alt.Tooltip("label:N", title="Class"),
                 alt.Tooltip("pct:Q", title="Share", format=".2%")]
    )
    ring = base.mark_arc(innerRadius=60)
    text = base.mark_text(radius=85, fontSize=14).encode(text=alt.Text("label:N"))
    return (ring + text).properties(title=title).configure_legend(orient="bottom")

def altair_bar(df: pd.DataFrame, title: str = "") -> alt.Chart:
    df = df.copy()
    df["pct"] = df["score"] / df["score"].sum()
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("label:N", title="Sentiment"),
        y=alt.Y("pct:Q", title="Probability", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("label:N", title="Class"),
                 alt.Tooltip("pct:Q", title="Probability", format=".2%")]
    ).properties(title=title).configure_legend(orient="bottom")

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model", list(MODEL_CHOICES.keys()), index=0)
    model_id = MODEL_CHOICES[model_name]

    st.caption(
        "Tip: CardiffNLP model provides **3 classes** (negative / neutral / positive). "
        "DistilBERT SST-2 has **2 classes** (negative / positive)."
    )

    st.write("—")
    st.markdown("**Batch Mode (CSV)**")
    up = st.file_uploader("Upload CSV with a text column", type=["csv"], help="UTF-8 CSV")
    text_col = st.text_input("Text column name", value="text", placeholder="e.g., review, comment, sentence")
    st.caption("We’ll add `pred_label`, `pred_score`, and per-class probabilities.")

# ==========================
# Header & Quick Examples
# ==========================
st.title("📝 Smart Sentiment Analyzer — Pro")
st.subheader("Analyze single texts or entire CSVs with a fast, friendly UI.")

samples = [
    "I love this product! It works perfectly. 😍",
    "This is the worst experience ever. 😡",
    "It's okay, not too bad but not great either. 🤔",
]
with st.container():
    st.markdown("**Quick Examples**")
    cols = st.columns(len(samples))
    for i, s in enumerate(samples):
        if cols[i].button(f"▶ {s[:26]}…"):
            st.session_state.user_input = s

# ==========================
# Load Model (cached)
# ==========================
try:
    classifier = load_model(model_id)
except Exception as e:
    st.error(f"Failed to load model `{model_id}`. Details: {e}")
    st.stop()

# ==========================
# Single Text Mode
# ==========================
st.markdown("### ✍️ Single Text")
user_input = st.text_area("Enter text to analyze", key="user_input", height=120, placeholder="Type or paste your sentence here…")

if user_input and user_input.strip():
    with st.spinner("Thinking…"):
        try:
            # IMPORTANT: return_all_scores=True -> fixes pie/bar by giving all class scores
            result_all = classifier(user_input, return_all_scores=True)
            # For a single input, pipeline returns [ [ {label,score}, ... ] ]
            scores = result_all[0] if isinstance(result_all, list) and isinstance(result_all[0], list) else result_all
            top = pick_top_label(scores)
            top_label = normalize_label(top["label"])
            top_score = float(top["score"])
            df_scores = scores_to_df(scores)

        except Exception as e:
            st.error(f"Could not analyze the text. Details: {e}")
            st.stop()

    # Nice summary row
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Overall Sentiment**")
        st.markdown(styled_badge(top_label), unsafe_allow_html=True)
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Confidence", f"{top_score:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("**Class Probabilities**")
        donut = altair_donut(df_scores, title="Probability Share (Donut)")
        bar = altair_bar(df_scores, title="Probability by Class (Bar)")
        st.altair_chart(donut, use_container_width=True)
        st.altair_chart(bar, use_container_width=True)

    with st.expander("🔍 Raw Model Output"):
        st.json(scores)

else:
    st.info("Enter text above or click a **Quick Example** to get started.")

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ==========================
# Batch CSV Mode
# ==========================
st.markdown("### 🧾 Batch Analyze (CSV)")
if up is not None:
    try:
        df_in = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if text_col not in df_in.columns:
        st.warning(f"Column `{text_col}` not found in the uploaded CSV. Available: {list(df_in.columns)}")
    else:
        st.success(f"Loaded {len(df_in)} rows. Click **Run Batch** to analyze.")
        if st.button("🚀 Run Batch"):
            results = []
            progress = st.progress(0)
            with st.spinner("Analyzing…"):
                for i, txt in enumerate(df_in[text_col].astype(str).tolist()):
                    try:
                        out = classifier(txt, return_all_scores=True)
                        sc = out[0] if isinstance(out, list) and isinstance(out[0], list) else out
                        best = pick_top_label(sc)
                        best_label = normalize_label(best["label"])
                        best_score = float(best["score"])
                        # Convert scores to dict with canonical keys
                        row_scores = {normalize_label(d["label"]): float(d["score"]) for d in sc}
                        results.append({
                            "pred_label": best_label,
                            "pred_score": best_score,
                            **row_scores
                        })
                    except Exception as e:
                        results.append({"pred_label": "error", "pred_score": np.nan})
                    if (i + 1) % max(1, len(df_in)//100) == 0:
                        progress.progress(min(1.0, (i + 1) / len(df_in)))
                progress.progress(1.0)

            df_out = pd.concat([df_in.reset_index(drop=True), pd.DataFrame(results)], axis=1)
            st.dataframe(df_out.head(50), use_container_width=True)

            # Distribution chart over predictions
            counts = df_out["pred_label"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
            dist = alt.Chart(counts).mark_bar().encode(
                x=alt.X("label:N", sort=ORDER),
                y=alt.Y("count:Q"),
                tooltip=["label", "count"]
            ).properties(title="Prediction Count by Class")
            st.altair_chart(dist, use_container_width=True)

            # Download
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Results (CSV)", csv, file_name="sentiment_results.csv", mime="text/csv")

# ==========================
# Footer
# ==========================
st.caption("Built with 🧠 Transformers + Streamlit. GPU detected: **{}**".format(torch.cuda.is_available()))
