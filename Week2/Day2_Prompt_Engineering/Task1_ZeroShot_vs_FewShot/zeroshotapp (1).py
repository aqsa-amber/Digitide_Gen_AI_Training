# app.py
# ------------------------------------------------------------
# Streamlit UI/UX upgrade for: Zero-shot vs Few-shot Sentiment
# ------------------------------------------------------------
# Highlights
# - Modern layout with sidebar controls and tabs
# - Single input or batch comparison (multi-line)
# - Editable few-shot examples
# - Color-coded results + confidence meters
# - PDF & CSV export
# - Robust prompt building and parsing
# ------------------------------------------------------------

import os
import io
import re
import tempfile
from typing import List, Tuple, Dict

import streamlit as st
import pandas as pd
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# Page config & custom styles
# -----------------------------
st.set_page_config(
    page_title="Zero-shot vs Few-shot Sentiment",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .chip {display:inline-block; padding:6px 12px; border-radius:18px; font-weight:600;}
      .chip-pos {background:#E6F7EA; color:#1D7A2E; border:1px solid #B9E6C5;}
      .chip-neg {background:#FDEDEE; color:#B00020; border:1px solid #F5C2C7;}
      .subtitle {opacity:0.8}
      .metric-box {border:1px solid #eee; border-radius:16px; padding:16px}
      .small-note {font-size:0.9rem; opacity:0.7}
      .footer {margin-top:24px; font-size:0.85rem; opacity:0.65}
      .codebox {background:#0f172a; color:#e2e8f0; padding:12px; border-radius:12px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("⚙️ Settings")

with st.sidebar.expander("Models", expanded=True):
    zs_model_name = st.text_input("Zero-shot model", value="valhalla/distilbart-mnli-12-1")
    fs_model_name = st.text_input("Few-shot (text2text) model", value="google/flan-t5-base")

with st.sidebar.expander("Labels", expanded=True):
    # For assignment parity we keep Positive/Negative, but you can add more.
    candidate_labels = ["positive", "negative"]
    st.write("Using labels:", ", ".join(candidate_labels))

with st.sidebar.expander("Few-shot Examples (editable)", expanded=True):
    ex1 = st.text_input("Example 1 (Positive)", value="I love this movie, it was amazing!")
    ex2 = st.text_input("Example 2 (Negative)", value="This food tastes terrible and I hate it.")
    ex3_show = st.checkbox("Add Example 3", value=True)
    ex3 = "I am happy with the service." if ex3_show else ""
    if ex3_show:
        ex3 = st.text_input("Example 3 (Positive/Negative)", value=ex3)

with st.sidebar.expander("PDF Export", expanded=False):
    include_rationales = st.checkbox("Include few-shot rationales in PDF", value=True)

st.sidebar.caption("Tip: For best results, run both methods on the same sentence(s) and compare.")

# -----------------------------
# Caching model loaders
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_zero_shot(model_name: str):
    return pipeline("zero-shot-classification", model=model_name)

@st.cache_resource(show_spinner=False)
def load_few_shot(model_name: str):
    return pipeline("text2text-generation", model=model_name)

# Initialize once
try:
    zero_shot_classifier = load_zero_shot(zs_model_name)
    few_shot_model = load_few_shot(fs_model_name)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def make_fewshot_prompt(sentence: str, examples: List[str]) -> str:
    # Determine simple polarity tag based on wording for examples (best-effort UX hint only)
    def tag_hint(text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["love", "amazing", "great", "happy", "good", "satisfied"]):
            return "Positive"
        if any(w in t for w in ["hate", "terrible", "bad", "awful", "sad", "worse", "worst"]):
            return "Negative"
        return "(label)"

    lines = [
        "Decide whether the sentiment of the sentence is Positive or Negative.",
        "Respond in the format: `<Label> - <brief justification>`.\n",
    ]

    for i, ex in enumerate(examples, start=1):
        if ex.strip():
            lines.append(f"Example {i}:\nSentence: {ex}\nAnswer: {tag_hint(ex)} - …\n")

    lines.append(f"Sentence: {sentence}\nAnswer:")
    return "\n".join(lines)


def parse_label_from_text(text: str) -> str:
    # Returns 'Positive' or 'Negative' if detected; else raw text head
    m = re.search(r"(Positive|Negative)", text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    # fallback: first token-like word
    head = text.strip().split("\n")[0]
    head = head.split("-")[0].strip()
    return head[:32]


def zero_shot_sentiment(sentence: str) -> Tuple[str, float]:
    result = zero_shot_classifier(sentence, candidate_labels=candidate_labels)
    # result["labels"] already sorted by score
    return result["labels"][0].capitalize(), float(result["scores"][0])


def few_shot_sentiment(sentence: str, examples: List[str]) -> Tuple[str, str]:
    prompt = make_fewshot_prompt(sentence, examples)
    out = few_shot_model(prompt, max_length=80, do_sample=False)[0]["generated_text"].strip()
    label = parse_label_from_text(out)
    return label, out


def label_chip(label: str) -> str:
    l = label.strip().lower()
    if l.startswith("pos"):
        return '<span class="chip chip-pos">Positive</span>'
    if l.startswith("neg"):
        return '<span class="chip chip-neg">Negative</span>'
    return f'<span class="chip">{label}</span>'


def generate_pdf(rows: List[Dict], title: str = "Sentiment Classification Report") -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 12))

    data = [["Sentence", "Zero-shot", "Confidence", "Few-shot (Label)", "Few-shot (Rationale)"]]
    for r in rows:
        data.append([
            r["sentence"],
            r["zs_label"],
            f"{r['zs_conf']:.2f}",
            r["fs_label"],
            r["fs_text"] if include_rationales else "—",
        ])

    table = Table(data, colWidths=[210, 80, 70, 100, 260])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#111827')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#CBD5E1')),
    ]))

    elements.append(Paragraph("Results", styles['Heading2']))
    elements.append(table)

    doc.build(elements)
    return temp_file.name


# -----------------------------
# Header
# -----------------------------
st.title("📊 Zero-shot vs Few-shot: Sentiment Analysis")
st.markdown("<p class='subtitle'>Compare a zero-shot classifier against a few-shot text2text model, then export your findings.</p>", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab_single, tab_batch, tab_about = st.tabs(["🔍 Single Input", "🧪 Batch Compare", "ℹ️ About & How it works"])

# -----------------------------
# Tab 1: Single Input
# -----------------------------
with tab_single:
    c1, c2 = st.columns([1, 1])
    with c1:
        sentence = st.text_area("Enter a sentence", height=120, placeholder="e.g., I love my new phone!")
        st.caption("Tip: Try ambiguous lines like ‘The weather is okay, not great.’")
        run_single = st.button("Run both methods", type="primary")

    with c2:
        st.markdown("**Few-shot examples used**")
        examples = [ex1, ex2]
        if ex3_show and ex3.strip():
            examples.append(ex3)
        st.markdown("\n".join([f"- {e}" for e in examples]) or "- (none)")

    if run_single and sentence.strip():
        with st.spinner("Analyzing…"):
            zs_label, zs_conf = zero_shot_sentiment(sentence)
            fs_label, fs_text = few_shot_sentiment(sentence, examples)

        m1, m2, m3 = st.columns([1,1,2])
        with m1:
            st.markdown("**Zero-shot**")
            st.markdown(label_chip(zs_label), unsafe_allow_html=True)
            st.progress(min(max(zs_conf, 0.0), 1.0))
            st.caption(f"Confidence: {zs_conf:.2f}")
        with m2:
            st.markdown("**Few-shot**")
            st.markdown(label_chip(fs_label), unsafe_allow_html=True)
        with m3:
            st.markdown("**Few-shot rationale**")
            st.markdown(f"<div class='codebox'>{fs_text}</div>", unsafe_allow_html=True)

        # Export row and allow PDF/CSV download
        row = {
            "sentence": sentence,
            "zs_label": zs_label.capitalize(),
            "zs_conf": zs_conf,
            "fs_label": fs_label.capitalize(),
            "fs_text": fs_text,
        }

        pdf_path = generate_pdf([row])
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF report", data=f, file_name="sentiment_report.pdf", mime="application/pdf")
        os.remove(pdf_path)

        csv_buf = io.StringIO()
        pd.DataFrame([row]).to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download CSV", data=csv_buf.getvalue(), file_name="sentiment_results.csv", mime="text/csv")

# -----------------------------
# Tab 2: Batch Compare
# -----------------------------
with tab_batch:
    st.write("Paste multiple sentences (one per line). We'll run both methods and compare side-by-side.")
    batch_text = st.text_area("Sentences (one per line)", height=180, placeholder="I love my new phone.\nThe movie was amazing.\nI don't like the service.\nThe weather is okay, not great.")

    go_batch = st.button("Run batch comparison", type="primary")

    if go_batch and batch_text.strip():
        sentences = [s.strip() for s in batch_text.split("\n") if s.strip()]
        rows = []
        with st.spinner("Running models on your sentences…"):
            for s in sentences:
                zs_label, zs_conf = zero_shot_sentiment(s)
                fs_label, fs_text = few_shot_sentiment(s, [ex1, ex2] + ([ex3] if ex3_show and ex3.strip() else []))
                rows.append({
                    "sentence": s,
                    "zs_label": zs_label.capitalize(),
                    "zs_conf": zs_conf,
                    "fs_label": fs_label.capitalize(),
                    "fs_text": fs_text,
                })

        df = pd.DataFrame(rows)
        # Display pretty dataframe
        st.dataframe(df, use_container_width=True)

        # Quick tallies
        cpos = sum(1 for r in rows if r["fs_label"].lower().startswith("pos"))
        cneg = sum(1 for r in rows if r["fs_label"].lower().startswith("neg"))
        zpos = sum(1 for r in rows if r["zs_label"].lower().startswith("pos"))
        zneg = sum(1 for r in rows if r["zs_label"].lower().startswith("neg"))

        colA, colB, colC, colD = st.columns(4)
        with colA:
            st.markdown("<div class='metric-box'>🔹 <b>Zero-shot Positive</b><br>"+str(zpos)+"</div>", unsafe_allow_html=True)
        with colB:
            st.markdown("<div class='metric-box'>🔹 <b>Zero-shot Negative</b><br>"+str(zneg)+"</div>", unsafe_allow_html=True)
        with colC:
            st.markdown("<div class='metric-box'>🔸 <b>Few-shot Positive</b><br>"+str(cpos)+"</div>", unsafe_allow_html=True)
        with colD:
            st.markdown("<div class='metric-box'>🔸 <b>Few-shot Negative</b><br>"+str(cneg)+"</div>", unsafe_allow_html=True)

        # Export buttons
        pdf_path = generate_pdf(rows, title="Batch Sentiment Comparison Report")
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF report", data=f, file_name="sentiment_batch_report.pdf", mime="application/pdf")
        os.remove(pdf_path)

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download CSV", data=csv_buf.getvalue(), file_name="sentiment_batch_results.csv", mime="text/csv")

# -----------------------------
# Tab 3: About
# -----------------------------
with tab_about:
    st.subheader("What this app demonstrates")
    st.markdown(
        """
        - **Zero-shot** uses an NLI-based classifier to map your sentence directly to labels without prior examples.
        - **Few-shot** feeds a small set of labeled examples to a text2text model (e.g., FLAN-T5) to steer outputs.
        - You can **edit examples** in the sidebar and observe how outputs shift.
        - Export **PDF** or **CSV** to include in your assignment deliverables.
        """
    )

    st.subheader("Reproducibility notes")
    st.markdown(
        """
        - Models are cached to speed up repeated runs (`@st.cache_resource`).\n
        - You can change model names in the sidebar if you prefer alternatives that exist on the HF Hub.\n
        - Few-shot prompt format is consistent: `<Label> - <brief justification>`. The app extracts labels robustly.
        """
    )

    st.subheader("How to run locally")
    st.code(
        """
        pip install streamlit transformers torch reportlab
        streamlit run app.py
        """,
        language="bash",
    )

st.markdown("<div class='footer'>Built for your Assignment 1 – polish the UI/UX, keep the core logic. ✨</div>", unsafe_allow_html=True)
