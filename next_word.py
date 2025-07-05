import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import zipfile
import gdown

# --- Page Configuration ---
st.set_page_config(
    page_title="FlowMate - AI Typing Assistant",
    page_icon="📝",
    layout="centered"
)

# --- Title and Tagline ---
st.markdown("<h1 class='title'>💬 FlowMate</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555; font-size:18px;'>Let your ideas flow. <em>We'll fill the words.</em></p>", unsafe_allow_html=True)

# --- Global Styles ---
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background: linear-gradient(to right, #e0eafc, #cfdef3);
    }

    .title {
        text-align: center;
        color: #5D3FD3;
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.5em;
        margin-top: 10px;
    }

    .feature-boxes {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 20px;
        margin-top: 10px;
        margin-bottom: 40px;
    }
    .feature {
        flex: 1 1 220px;
        padding: 20px;
        border-radius: 12px;
        color: #222;
        font-size: 17px;
        font-weight: 500;
        background-color: #fefefe;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: left;
    }
    .feature span {
        font-weight: bold;
        font-size: 19px;
        display: block;
        margin-bottom: 6px;
        color: #2C3E50;
    }

    .stTextArea textarea {
        font-size: 18px !important;
        padding: 14px !important;
        border-radius: 12px !important;
        color: #000 !important;
        background-color: #ffffff !important;
    }

    .footer-text {
        text-align: center;
        font-size: 0.9em;
        color: #777;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Intro Text ---
st.markdown("<p style='text-align:center; color:#555;'>Smart typing assistant that predicts your next word in real-time!</p>", unsafe_allow_html=True)

# --- Feature Banner ---
st.markdown("<h3 style='text-align:center; color:#5D3FD3;'>🚀 Why FlowMate?</h3>", unsafe_allow_html=True)
st.markdown("""
<div class="feature-boxes">
    <div class="feature" style="background-color:#FFE6F0;">
        <span>💡 Smart Suggestions</span>
        Instantly suggests words to help you write better, faster, and more creatively.
    </div>
    <div class="feature" style="background-color:#E6F7FF;">
        <span>📝 Creative Boost</span>
        Use it while writing stories, blogs, or notes — it gives you that extra spark!
    </div>
    <div class="feature" style="background-color:#EAFFEA;">
        <span>⚡ Just Click & Go</span>
        It’s a simple one-click tool — no login, no ads, just fun writing!
    </div>
    <div class="feature" style="background-color:#FFF3CD;">
        <span>📱 Works Everywhere</span>
        Whether you’re on a laptop or phone, FlowMate works beautifully.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Load Fine-tuned GPT-2 Model from Google Drive ZIP ---
MODEL_URL = "https://drive.google.com/uc?id=1lUu0_3gnxZ90xfEmujVIO73eLm99FsdB"
MODEL_ZIP = "gpt2-wikitext-best-model.zip"
MODEL_DIR = "gpt2-wikitext-best-model"

@st.cache_resource
def load_finetuned_model():
    if not os.path.exists(MODEL_DIR):
        with st.spinner("📦 Downloading fine-tuned GPT-2 model..."):
            gdown.download(MODEL_URL, MODEL_ZIP, quiet=False)

        with st.spinner("📂 Extracting model..."):
            with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
                zip_ref.extractall()

    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to("cpu")
    model.eval()
    return model, tokenizer

model, tokenizer = load_finetuned_model()

# --- Session State ---
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "predicted_words" not in st.session_state:
    st.session_state.predicted_words = []

# --- Input Area ---
st.markdown("### 💬 Start Typing Below")
text = st.text_area(
    "", value=st.session_state.input_text, key="user_input_area", height=120,
    placeholder="e.g., The day was so beautiful that..."
)

# --- Prediction Button ---
if st.button("✨ Suggest a Word!"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        st.session_state.input_text = text
        max_len = model.config.max_position_embeddings - 1
        encoded_input = tokenizer.encode(st.session_state.input_text, return_tensors="pt")
        input_ids = encoded_input[:, -max_len:] if encoded_input.shape[1] > max_len else encoded_input

        input_ids = input_ids.to("cpu")
        outputs = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 1,
            num_return_sequences=5,
            do_sample=True,
            top_k=50,
            temperature=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

        predictions = set()
        current_len = input_ids.shape[1]

        for output in outputs:
            if output.shape[0] > current_len:
                next_token_id = output[current_len].item()
                next_word = tokenizer.decode(next_token_id, skip_special_tokens=True).strip()
                if next_word:
                    predictions.add(next_word.split(' ')[0])

        st.session_state.predicted_words = list(predictions)[:3]

# --- Prediction Buttons ---
if st.session_state.predicted_words:
    st.markdown("### 🧠 Suggested Words")
    st.markdown("<p style='color:#333;'>Click a word to continue writing:</p>", unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.predicted_words))
    for i, word in enumerate(st.session_state.predicted_words):
        if cols[i].button(f"👉 {word}"):
            st.session_state.input_text = text + " " + word
            st.session_state.predicted_words = []
            st.rerun()

# --- Keep input synced ---
st.session_state.input_text = text

# --- Footer ---
st.markdown("""<hr style="margin-top:30px;">
<center class="footer-text">
<p>💬 Try typing: <i>\"Once upon a time\"</i>, <i>\"The world is\"</i>, or <i>\"AI will\"</i></p>
<p>Built with ❤️ using Streamlit & Transformers</p>
</center>""", unsafe_allow_html=True)
