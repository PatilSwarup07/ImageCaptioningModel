"""
app.py – Streamlit Image Captioning Interface
Xception + Attention LSTM  |  Flickr8k
Designed for LOCAL VS Code use after training in Google Colab
"""

import streamlit as st
from PIL import Image
import os, time

st.set_page_config(
    page_title="VisionWords – Image Captioning",
    page_icon="🔭",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0d0d0f;
    color: #e8e4dc;
}
.vw-header { padding:2.5rem 0 1rem; text-align:center; border-bottom:1px solid #2a2a30; margin-bottom:2rem; }
.vw-header h1 { font-family:'DM Serif Display',serif; font-size:3rem; letter-spacing:-0.02em;
    background:linear-gradient(135deg,#f5c842 0%,#f08533 60%,#e0446e 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin:0; }
.vw-header p { color:#888; font-size:0.95rem; font-weight:300; letter-spacing:0.1em; text-transform:uppercase; margin-top:0.4rem; }
.upload-hint { text-align:center; color:#555; font-size:0.85rem; margin-top:0.5rem;
    font-family:'DM Mono',monospace; letter-spacing:0.05em; }
.caption-card { background:linear-gradient(135deg,#1a1a22 0%,#161620 100%);
    border:1px solid #2e2e3a; border-radius:16px; padding:1.8rem 2rem; margin-top:1.5rem; position:relative; overflow:hidden; }
.caption-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#f5c842,#f08533,#e0446e); }
.caption-label { font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:0.18em;
    text-transform:uppercase; color:#f08533; margin-bottom:0.75rem; }
.caption-text { font-family:'DM Serif Display',serif; font-size:1.55rem; line-height:1.4; color:#f0ece4; font-style:italic; }
.caption-meta { margin-top:1.2rem; display:flex; gap:1.2rem; flex-wrap:wrap; }
.meta-badge { font-family:'DM Mono',monospace; font-size:0.72rem; color:#666;
    background:#1f1f28; border:1px solid #2d2d38; border-radius:6px; padding:0.3rem 0.7rem; letter-spacing:0.06em; }
.info-box { background:#111116; border:1px solid #252530; border-radius:10px; padding:1rem 1.2rem;
    font-size:0.82rem; color:#aaa; margin-bottom:1rem; font-family:'DM Mono',monospace; line-height:1.8; }
.step-box { background:#0f1218; border:1px solid #1e2530; border-left:3px solid #f08533;
    border-radius:8px; padding:1rem 1.2rem; margin:0.5rem 0; font-size:0.82rem;
    font-family:'DM Mono',monospace; line-height:1.8; color:#ccc; }
.status-ok  { color:#4ade80; }
.status-err { color:#f87171; }
.stButton > button { background:linear-gradient(135deg,#f5c842,#f08533) !important;
    color:#0d0d0f !important; font-family:'DM Sans',sans-serif !important; font-weight:500 !important;
    border:none !important; border-radius:10px !important; padding:0.6rem 2rem !important;
    font-size:0.95rem !important; letter-spacing:0.03em !important; transition:opacity 0.2s !important; }
.stButton > button:hover { opacity:0.88 !important; }
.stSelectbox label, .stSlider label, .stRadio label {
    color:#aaa !important; font-size:0.82rem !important; font-family:'DM Mono',monospace !important;
    letter-spacing:0.05em !important; text-transform:uppercase !important; }
div[data-testid="stFileUploaderDropzone"] { background:#111116 !important;
    border:2px dashed #2e2e3a !important; border-radius:14px !important; }
footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="vw-header">
  <h1>VisionWords</h1>
  <p>Xception · Attention LSTM · Flickr8k</p>
</div>
""", unsafe_allow_html=True)

# ── Check files ──────────────────────────────────────────────────────
try:
    from inference import check_model_files
    file_status = check_model_files()
except Exception:
    file_status = {
        "captioning_model.keras": os.path.exists("captioning_model_final.keras"),
        "tokenizer.pkl":          os.path.exists("tokenizer_final.pkl"),
        "config.pkl":             os.path.exists("config_final.pkl"),
    }

model_ready = all(file_status.values())

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    decode_method = st.radio("Decoding method", ["Beam Search", "Greedy"], index=0)
    beam_width = 3
    if decode_method == "Beam Search":
        beam_width = st.slider("Beam width", 1, 10, 3)

    st.markdown("---")
    st.markdown("### 📂 Model Files")
    for fname, exists in file_status.items():
        icon = "✅" if exists else "❌"
        cls  = "status-ok" if exists else "status-err"
        st.markdown(
            f"<span class='{cls}'>{icon}</span> `{fname}`",
            unsafe_allow_html=True
        )

    if not model_ready:
        st.markdown("---")
        st.markdown("### 🚀 Setup Guide")
        st.markdown("""
<div class="step-box">
<b>Step 1 — Train in Colab</b><br>
Open <code>ImageCaptioning_Colab.ipynb</code><br>
Runtime → T4 GPU → Run All
</div>
<div class="step-box">
<b>Step 2 — Download 3 files</b><br>
Run the last Colab cell:<br>
• captioning_model.keras<br>
• tokenizer.pkl<br>
• config.pkl
</div>
<div class="step-box">
<b>Step 3 — Place in project root</b><br>
Same folder as <code>app.py</code>:<br><br>
your-repo/<br>
├── app.py<br>
├── inference.py<br>
├── captioning_model.keras ✅<br>
├── tokenizer.pkl &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅<br>
└── config.pkl &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅
</div>
<div class="step-box">
<b>Step 4 — Relaunch</b><br>
<code>streamlit run app.py</code>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏗 Architecture")
    st.markdown("""
<div class="info-box">
<b>Extractor</b> &nbsp;→ Xception (ImageNet)<br>
<b>Features</b> &nbsp;&nbsp;→ GAP → 2048-d<br>
<b>Decoder</b> &nbsp;&nbsp;&nbsp;→ Embedding + LSTM(512)<br>
<b>Attention</b> → Scaled dot-product<br>
<b>Dataset</b> &nbsp;&nbsp;&nbsp;→ Flickr8k (8k images)
</div>
""", unsafe_allow_html=True)

# ── Main area — show setup card if files missing ─────────────────────
if not model_ready:
    missing = [k for k, v in file_status.items() if not v]
    st.warning(
        f"**{len(missing)} file(s) missing:** `{'`, `'.join(missing)}`\n\n"
        "Follow the **Setup Guide** in the sidebar to train in Google Colab "
        "and download the required files."
    )
    st.info(
        "💡 **Quick start:** Open `ImageCaptioning_Colab.ipynb` in Google Colab, "
        "enable a T4 GPU, and run all cells. Training takes ~15 minutes. "
        "The last cell will auto-download the 3 files you need."
    )

# ── Upload ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)
st.markdown('<p class="upload-hint">JPG · PNG · WEBP  ·  any size</p>',
            unsafe_allow_html=True)

# ── Generate ──────────────────────────────────────────────────────────
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        #st.image(image, use_container_width=True)
        st.image(image, use_column_width=True)


    with col2:
        w, h = image.size
        st.markdown(f"""
<div class="info-box">
<b>name</b> &nbsp;→ {uploaded.name}<br>
<b>size</b> &nbsp;→ {uploaded.size // 1024} KB<br>
<b>dims</b> &nbsp;→ {w} × {h} px<br>
<b>decode</b> → {decode_method.lower()}{f' (k={beam_width})' if decode_method=='Beam Search' else ''}
</div>
""", unsafe_allow_html=True)
        generate_btn = st.button("✨ Generate Caption", disabled=not model_ready)

    if generate_btn:
        with st.spinner("Analysing image …"):
            try:
                from inference import caption_image
                t0      = time.time()
                method  = "beam" if decode_method == "Beam Search" else "greedy"
                caption = caption_image(image, method=method, beam_width=beam_width)
                elapsed = time.time() - t0

                st.markdown(f"""
<div class="caption-card">
  <div class="caption-label">Generated Caption</div>
  <div class="caption-text">"{caption}"</div>
  <div class="caption-meta">
    <span class="meta-badge">⏱ {elapsed:.2f}s</span>
    <span class="meta-badge">🔤 {len(caption.split())} words</span>
    <span class="meta-badge">🔭 xception + attn-lstm</span>
  </div>
</div>
""", unsafe_allow_html=True)
                st.text_area("Copy caption", caption, height=80,
                             label_visibility="collapsed")

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.exception(e)

st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#444;font-size:0.75rem;
          font-family:"DM Mono",monospace;letter-spacing:0.08em;'>
  VISIONWORDS · XCEPTION + ATTENTION LSTM · FLICKR8K · TENSORFLOW / KERAS
</p>
""", unsafe_allow_html=True)
