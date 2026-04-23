import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    .hero-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
    }

    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }

    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 0.75rem;
    }

    .summary-text {
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.8;
    }

    .bullet-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        color: #e2e8f0;
        font-size: 0.97rem;
        line-height: 1.6;
    }

    .bullet-item:last-child {
        border-bottom: none;
    }

    .bullet-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        margin-top: 0.45rem;
        flex-shrink: 0;
    }

    .stat-box {
        background: rgba(167, 139, 250, 0.1);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        text-align: center;
    }

    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #a78bfa;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.2) !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
    }

    .stSlider > div > div {
        color: #a78bfa !important;
    }

    label, .stSlider label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
    }

    .reduction-badge {
        display: inline-block;
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


# ── Core functions ─────────────────────────────────────────────────────────────
def summarize_text(text, tokenizer, model, max_len, min_len):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def to_bullet_points(summary, n=3):
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= n:
        step = len(sentences) / n
        picked = [sentences[int(i * step)] for i in range(n)]
    else:
        picked = sentences
        while len(picked) < n:
            picked.append(picked[-1])
    return picked


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">📝 AI Text Summarizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Powered by BART · Paste any article and get an instant summary with key takeaways</p>', unsafe_allow_html=True)

# Load model
with st.spinner("⚡ Loading AI model (first run may take a moment)..."):
    tokenizer, model = load_model()

# Layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-title">📄 Input Text</div>', unsafe_allow_html=True)
    input_text = st.text_area(
        label="input_text",
        label_visibility="collapsed",
        placeholder="Paste your article, blog post, or any long text here...",
        height=320,
    )

    word_count = len(input_text.split()) if input_text.strip() else 0
    st.caption(f"📊 Word count: **{word_count}**")

    with st.expander("⚙️ Advanced Settings"):
        max_length = st.slider("Max summary length (tokens)", 60, 300, 150, step=10)
        min_length = st.slider("Min summary length (tokens)", 20, 100, 40, step=5)
        num_bullets = st.slider("Number of bullet points", 2, 5, 3)

    summarize_btn = st.button("✨ Summarize", use_container_width=True)

with col_right:
    if summarize_btn:
        if not input_text.strip():
            st.warning("⚠️ Please paste some text first.")
        elif word_count < 30:
            st.warning("⚠️ Text is too short. Please paste at least 30 words.")
        else:
            with st.spinner("🤖 Generating summary..."):
                summary = summarize_text(input_text, tokenizer, model, max_length, min_length)
                bullets = to_bullet_points(summary, n=num_bullets)

            summary_words = len(summary.split())
            reduction = round((1 - summary_words / word_count) * 100)

            # Stats row
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f'<div class="stat-box"><div class="stat-value">{word_count}</div><div class="stat-label">Original Words</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="stat-box"><div class="stat-value">{summary_words}</div><div class="stat-label">Summary Words</div></div>', unsafe_allow_html=True)
            with s3:
                st.markdown(f'<div class="stat-box"><div class="stat-value">{reduction}%</div><div class="stat-label">Reduced By</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Summary card
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📋 Summary</div>
                <div class="summary-text">{summary}</div>
            </div>
            """, unsafe_allow_html=True)

            # Bullet points card
            bullets_html = "".join(
                f'<div class="bullet-item"><div class="bullet-dot"></div><div>{b}</div></div>'
                for b in bullets
            )
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🎯 Key Takeaways</div>
                {bullets_html}
            </div>
            """, unsafe_allow_html=True)

            # Copy area
            with st.expander("📋 Copy Summary Text"):
                st.text_area("summary_copy", value=summary, label_visibility="collapsed", height=100)

    else:
        # Placeholder state
        st.markdown("""
        <div class="card" style="text-align:center; padding: 4rem 2rem;">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">🧠</div>
            <div style="color: #94a3b8; font-size: 1rem; line-height: 1.8;">
                Paste your text on the left<br>and click <strong style="color:#a78bfa">Summarize</strong> to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
