import os
import time
import json
import threading
from typing import List, Optional

import streamlit as st

# --- Optional GPU/CPU libs (torch used only for device detection & tensors) ---
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    logging as hf_logging,
)

# Silence HF warnings for cleaner UI
hf_logging.set_verbosity_error()

# ========================
# Page / Theming
# ========================
st.set_page_config(
    page_title="🧠 Local LLM Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for gradient background + glassmorphism
st.markdown(
    """
    <style>
      /* Background gradient */
      .stApp {
        background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 50%, #22c55e 100%);
        background-attachment: fixed;
      }
      /* Glass cards */
      .glass {
        background: rgba(255,255,255,0.16);
        border-radius: 18px;
        padding: 1.25rem 1.25rem 0.75rem 1.25rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 10px 30px rgba(0,0,0,0.10);
      }
      /* Tweak widgets */
      .stTextArea textarea { border-radius: 14px !important; }
      .stButton>button { border-radius: 999px; padding: 0.6rem 1.1rem; font-weight: 600; }
      .metric-box { font-size: 0.9rem; opacity: 0.9; }
      .small-note { font-size: 0.85rem; opacity: 0.85; }
      code, pre { border-radius: 10px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================
# Utilities & State
# ========================

def get_device() -> str:
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except Exception:
        pass
    return "cpu"

DEVICE = get_device()

if "chat" not in st.session_state:
    st.session_state.chat: List[dict] = []  # [{role:"user"|"assistant", content:str}]

if "model_id" not in st.session_state:
    st.session_state.model_id = "gpt2"

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model" not in st.session_state:
    st.session_state.model = None

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None
if "last_tokens" not in st.session_state:
    st.session_state.last_tokens = None

# Popular small models (CPU-friendly first). You can add/remove.
POPULAR_MODELS = [
    ("GPT-2 (very small, CPU-friendly)", "gpt2"),
    ("DistilGPT-2 (small, CPU-friendly)", "distilgpt2"),
    ("TinyLlama 1.1B Chat (quant needed for CPU)", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("Phi-2 (requires some RAM)", "microsoft/phi-2"),
]

# ========================
# Model Loading (cached)
# ========================
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load tokenizer and model with sensible defaults. Cached across reruns."""
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tok.pad_token is None:
            # Ensure a pad token for generation
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if DEVICE != "cpu" else None,
            torch_dtype=(torch.float16 if DEVICE == "cuda" else None) if torch is not None else None,
        )
        if torch is not None:
            model.to(DEVICE)
        return tok, model, None
    except Exception as e:
        return None, None, str(e)

# ========================
# Sidebar Controls
# ========================
with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown(f"**Device:** `{DEVICE}`")

    # Model picker
    label_to_id = {label: mid for label, mid in POPULAR_MODELS}
    chosen_label = st.selectbox(
        "Choose a model",
        list(label_to_id.keys()),
        index=0,
        help="Pick a small model first to verify your setup."
    )
    default_id = label_to_id[chosen_label]

    manual_id = st.text_input(
        "Or enter a custom HF model id",
        value=default_id,
        help="Example: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    # Decoding controls
    st.subheader("Decoding")
    max_new_tokens = st.slider("Max new tokens", 16, 1024, 160, 8)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.8, 0.05)
    top_p = st.slider("Top-p (nucleus)", 0.1, 1.0, 0.95, 0.01)
    top_k = st.slider("Top-k", 0, 200, 50, 5)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.05, 0.01)

    seed = st.number_input("Seed (for reproducibility, -1 = random)", value=-1, step=1)

    # Stop sequences help UX for some chatty models
    stop_sequences_raw = st.text_input(
        "Stop sequences (comma-separated)", value="",
        help="Optional: e.g. \nUser:,\nAssistant:"
    )
    stop_sequences = [s.strip() for s in stop_sequences_raw.split(",") if s.strip()]

    st.markdown("---")
    enable_stream = st.toggle("Live streaming output", value=True)

    st.markdown("---")
    st.caption("Trouble with downloads? Some models require internet on first run to fetch weights. After that they're local.")

# ========================
# Header & Metrics
# ========================
colA, colB, colC = st.columns([2, 1, 1])
with colA:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.title("🧠 Local LLM Playground")
    st.write("Type a prompt, generate text, and measure latency — all locally.")
    st.markdown('</div>', unsafe_allow_html=True)

with colB:
    with st.container(border=True):
        st.metric("Last response (s)", f"{st.session_state.last_latency:.2f}" if st.session_state.last_latency else "–")
        st.caption("Time to finish generation")
with colC:
    with st.container(border=True):
        st.metric("Tokens generated", f"{st.session_state.last_tokens}" if st.session_state.last_tokens else "–")
        st.caption("Approx. new tokens")

# ========================
# Model Loader Bar
# ========================
with st.spinner("Loading model… this is cached for speed on reruns."):
    tokenizer, model, load_err = load_model(manual_id)

if load_err:
    st.error("❌ Failed to load model: " + str(load_err))
    st.stop()

# ========================
# Chat / Prompt Area
# ========================
left, right = st.columns([2, 1])
with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("💬 Chat")

    # Display history
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # Prompt box (chat-input style)
    user_prompt = st.chat_input("Type your prompt (e.g., 'Write a short poem about AI')…")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("✨ Examples")
    st.markdown(
        """
        - "Write a short poem about a robot learning to love."
        - "Explain AI as if I were 5 years old."
        - "Start a mystery story on a rainy night."
        - "Give me a funny programmer joke."
        - "Summarize supervised vs unsupervised learning."
        """
    )

    st.markdown("---")
    st.subheader("🧾 Transcript")
    if st.session_state.chat:
        if st.button("📥 Download chat as JSON"):
            fname = "chat_transcript.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(st.session_state.chat, f, ensure_ascii=False, indent=2)
            with open(fname, "rb") as f:
                st.download_button("Download", data=f, file_name=fname, mime="application/json")
    else:
        st.caption("No messages yet.")

    st.markdown("---")
    if st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# Generation Logic
# ========================

def _apply_stop_sequences(text: str, stops: List[str]) -> str:
    if not stops:
        return text
    cut = len(text)
    for s in stops:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def generate_stream(prompt: str) -> str:
    """Stream tokens with TextIteratorStreamer for a smoother UX."""
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Seed control
    if seed is not None and int(seed) >= 0 and torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, timeout=60.0)
    gen_kwargs = dict(
        **input_ids,
        max_new_tokens=int(max_new_tokens),
        do_sample=temperature > 0,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial_text = ""
    placeholder = st.empty()
    token_count = 0
    start = time.time()

    for new_text in streamer:
        partial_text += new_text
        token_count += 1
        safe_text = _apply_stop_sequences(partial_text, stop_sequences)
        placeholder.markdown(safe_text)
        # Early stop if stop sequence appears
        if safe_text != partial_text:
            break

    latency = time.time() - start
    st.session_state.last_latency = latency
    st.session_state.last_tokens = token_count
    return _apply_stop_sequences(partial_text, stop_sequences)


def generate_blocking(prompt: str) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Seed control
    if seed is not None and int(seed) >= 0 and torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    start = time.time()
    output_ids = model.generate(
        **input_ids,
        max_new_tokens=int(max_new_tokens),
        do_sample=temperature > 0,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    latency = time.time() - start

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the newly generated continuation for clarity
    # (simple heuristic; for chat models, the full text can be okay)
    if text.startswith(prompt):
        text = text[len(prompt):]

    text = _apply_stop_sequences(text, stop_sequences)
    st.session_state.last_latency = latency
    st.session_state.last_tokens = int(max_new_tokens)
    return text

# Handle a new user prompt
if user_prompt:
    st.session_state.chat.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        try:
            if enable_stream:
                response = generate_stream(user_prompt)
            else:
                response = generate_blocking(user_prompt)
        except Exception as e:
            st.error(f"❌ Generation failed: {e}")
            st.info(
                """
                **Troubleshooting tips**
                - If you see CUDA/memory errors, switch to a smaller model (e.g., `gpt2`) or reduce *max_new_tokens*.
                - If downloads fail, ensure internet for the first run (to fetch model weights). Afterwards, it's fully local.
                - On Apple Silicon, try running with device `mps` (auto-detected if available).
                - On CPU, be patient with larger models — they can still work with lower *max_new_tokens*.
                """
            )
            response = ""
        if response:
            st.markdown(response)
            st.session_state.chat.append({"role": "assistant", "content": response})

# ========================
# Footer: System Info & Help
# ========================
st.markdown("---")

sys_col1, sys_col2 = st.columns(2)
with sys_col1:
    with st.container(border=True):
        st.subheader("🧩 Environment")
        st.markdown(
            f"""
            - **Device:** `{DEVICE}`
            - **Model:** `{manual_id}`
            - **Torch:** `{torch.__version__ if torch else 'not installed'}`
            - **Transformers:** displayed on import
            """
        )

with sys_col2:
    with st.container(border=True):
        st.subheader("🛠 Quick Start (VS Code)")
        st.markdown(
            """
            **Install libraries (CPU example):**
            ```bash
            pip install -U streamlit transformers accelerate
            pip install torch --index-url https://download.pytorch.org/whl/cpu
            ```
            **Run the app:**
            ```bash
            streamlit run enhanced_local_llm_app.py
            ```
            The terminal shows a **local URL** (e.g., http://localhost:8501). Click to open immediately.
            """
        )

st.caption(
    "Pro tip: The first time you select a model, it will download from the Hugging Face Hub. Subsequent runs are fully local."
)
