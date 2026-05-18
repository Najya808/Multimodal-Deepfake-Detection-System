import streamlit as st
from PIL import Image
import numpy as np
import os

from backend.face_detector import detect_faces
from backend.face_classifier import predict_face
from backend.video_pipeline import analyze_video


# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DeepFake Forensics AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ══════════════════════════════════════════════════════════
#  FULL CYBER UI STYLES
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

.stApp {
    background: radial-gradient(ellipse at 20% 10%, #1a0533 0%, #0a0a1a 40%, #000510 100%);
    min-height: 100vh;
}
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px !important; }

/* ── NAVBAR ── */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 0;
    border-bottom: 1px solid rgba(123,47,247,0.25);
    margin-bottom: 0;
}
.nav-logo {
    font-family: 'Orbitron', monospace;
    font-size: 20px;
    font-weight: 900;
    background: linear-gradient(90deg, #00f5ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
}
.nav-links { display: flex; gap: 28px; align-items: center; }
.nav-links a {
    color: rgba(255,255,255,0.65) !important;
    font-size: 13px; font-weight: 600;
    text-decoration: none !important;
    letter-spacing: 1px; text-transform: uppercase;
}
.nav-links a:hover { color: #00f5ff !important; }
.nav-btn {
    background: linear-gradient(135deg, #7b2ff7, #00f5ff22);
    border: 1px solid #7b2ff7;
    color: #fff !important;
    padding: 7px 18px; border-radius: 6px;
    font-size: 13px; font-weight: 700; letter-spacing: 1px;
}

/* ── HERO ── */
.hero-section {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 60px 0 50px 0;
    gap: 40px;
}
.hero-left { flex: 1; max-width: 540px; }
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,245,255,0.08);
    border: 1px solid rgba(0,245,255,0.25);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px; color: #00f5ff;
    font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 22px;
}
.hero-badge::before {
    content: '';
    width: 7px; height: 7px;
    background: #00f5ff; border-radius: 50%;
    box-shadow: 0 0 8px #00f5ff;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.4; transform:scale(1.3); }
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 48px; font-weight: 900;
    line-height: 1.15; margin: 0 0 10px 0;
    color: #ffffff; letter-spacing: 1px;
}
.hero-title-accent {
    font-family: 'Orbitron', monospace;
    font-size: 48px; font-weight: 900;
    background: linear-gradient(90deg, #00f5ff 0%, #7b2ff7 60%, #00ff85 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px; display: block;
}
.hero-sub {
    color: rgba(200,200,230,0.7);
    font-size: 16px; line-height: 1.7;
    margin: 16px 0 32px 0; max-width: 460px;
}

/* ── HERO VISUAL ── */
.hero-right {
    flex: 1; display: flex;
    justify-content: center; align-items: center;
    position: relative; min-height: 360px;
}
.iso-frame {
    width: 420px; height: 340px;
    background: radial-gradient(ellipse at 60% 40%, rgba(123,47,247,0.18), transparent 70%),
                radial-gradient(ellipse at 30% 70%, rgba(0,245,255,0.12), transparent 60%);
    border-radius: 24px;
    border: 1px solid rgba(123,47,247,0.3);
    display: flex; align-items: center; justify-content: center;
    position: relative; overflow: hidden;
}
.iso-frame::before {
    content: ''; position: absolute; inset: 0;
    background:
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(123,47,247,0.06) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,245,255,0.04) 40px);
}
.iso-inner { position: relative; z-index: 2; text-align: center; }
.iso-icon-main {
    font-size: 80px;
    filter: drop-shadow(0 0 30px rgba(0,245,255,.8)) drop-shadow(0 0 60px rgba(123,47,247,.5));
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-12px); }
}
.iso-rings { position: absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:200px; height:200px; }
.ring {
    position: absolute; border-radius: 50%; border: 1px solid;
    top:50%; left:50%; transform: translate(-50%,-50%);
}
.ring-1 { width:120px; height:120px; border-color:rgba(0,245,255,.3); animation: spin 6s linear infinite; }
.ring-2 { width:180px; height:180px; border-color:rgba(123,47,247,.25); animation: spin 10s linear infinite reverse; }
.ring-3 { width:240px; height:240px; border-color:rgba(0,255,133,.15); animation: spin 15s linear infinite; }
@keyframes spin { to { transform: translate(-50%,-50%) rotate(360deg); } }
.orb { position: absolute; border-radius: 50%; animation: floatOrb 4s ease-in-out infinite; }
.orb-1 { width:12px; height:12px; background:#00f5ff; top:18%; left:15%; box-shadow:0 0 15px #00f5ff; }
.orb-2 { width:8px;  height:8px;  background:#7b2ff7; top:72%; left:80%; box-shadow:0 0 12px #7b2ff7; animation-delay:1s; }
.orb-3 { width:10px; height:10px; background:#00ff85; top:80%; left:20%; box-shadow:0 0 12px #00ff85; animation-delay:2s; }
@keyframes floatOrb {
    0%,100% { transform:translate(0,0); }
    50%      { transform:translate(8px,-8px); }
}

/* ── STATS BAR ── */
.stats-bar {
    display: flex;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(123,47,247,0.2);
    border-radius: 14px;
    margin-bottom: 50px;
    overflow: hidden;
}
.stat-item {
    flex: 1; padding: 22px 20px; text-align: center;
    border-right: 1px solid rgba(123,47,247,0.15);
}
.stat-item:last-child { border-right: none; }
.stat-number {
    font-family: 'Orbitron', monospace; font-size: 28px; font-weight: 700;
    background: linear-gradient(90deg, #00f5ff, #7b2ff7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: block;
}
.stat-label { font-size: 12px; color: rgba(200,200,255,.5); font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }

/* ── SECTION HEADING ── */
.section-heading { font-family: 'Orbitron', monospace; font-size: 26px; font-weight: 700; color: #fff; margin-bottom: 6px; }
.section-sub { color: rgba(180,180,220,.6); font-size: 15px; margin-bottom: 24px; }

/* ── UPLOAD ZONE ── */
.upload-zone {
    background: rgba(123,47,247,0.06);
    border: 2px dashed rgba(123,47,247,0.4);
    border-radius: 16px; padding: 32px 24px;
    text-align: center; cursor: pointer;
    position: relative; overflow: hidden;
    transition: border-color .3s, background .3s;
}
.upload-zone:hover { border-color: rgba(0,245,255,.6); background: rgba(0,245,255,.04); }
.upload-icon { font-size: 44px; margin-bottom: 10px; }
.upload-title { font-family: 'Orbitron', monospace; font-size: 16px; font-weight: 700; color: #fff; margin-bottom: 4px; letter-spacing: 1px; }
.upload-hint { color: rgba(180,180,220,.5); font-size: 13px; }

/* ── INFO CARD ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(123,47,247,0.25);
    border-radius: 14px; padding: 20px 18px;
    position: relative; overflow: hidden;
    margin-bottom: 14px;
    transition: border-color .3s, box-shadow .3s;
}
.card:hover { border-color: rgba(0,245,255,.35); box-shadow: 0 0 28px rgba(0,245,255,.07); }
.card::after {
    content: ''; position: absolute;
    top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, #7b2ff7, #00f5ff, transparent);
    border-radius: 14px 14px 0 0;
}
.card-icon { font-size: 26px; margin-bottom: 8px; }
.card-title { font-family: 'Orbitron', monospace; font-size: 13px; font-weight: 700; color: #fff; letter-spacing: 1px; margin-bottom: 5px; }
.card-desc { font-size: 13px; color: rgba(180,180,220,.55); line-height: 1.6; }

/* ── RESULT CARDS ── */
.result-card {
    background: rgba(255,255,255,0.04);
    border-radius: 16px; padding: 26px 22px;
    border: 1px solid rgba(123,47,247,0.25);
    position: relative; overflow: hidden;
}
.result-card::after {
    content: ''; position: absolute;
    top:0; left:0; right:0; height:3px;
    border-radius: 16px 16px 0 0;
}
.result-card.fake-card  { border-color: rgba(255,50,80,.4); }
.result-card.fake-card::after  { background: linear-gradient(90deg, #ff3250, #ff8c00); }
.result-card.real-card  { border-color: rgba(0,255,133,.4); }
.result-card.real-card::after  { background: linear-gradient(90deg, #00ff85, #00f5ff); }

.result-label { font-family: 'Orbitron', monospace; font-size: 32px; font-weight: 900; letter-spacing: 2px; margin-bottom: 10px; }
.result-fake { color: #ff3250; text-shadow: 0 0 20px rgba(255,50,80,.6); }
.result-real { color: #00ff85; text-shadow: 0 0 20px rgba(0,255,133,.6); }

.result-meta { color: rgba(200,200,255,.6); font-size: 14px; line-height: 2; }
.result-meta span { color: #fff; font-weight: 600; }

/* ── CONFIDENCE BAR ── */
.conf-wrap { margin: 14px 0; }
.conf-row { display: flex; justify-content: space-between; font-size: 12px; color: rgba(200,200,255,.6); margin-bottom: 5px; }
.conf-track { height: 6px; background: rgba(255,255,255,.08); border-radius: 6px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #7b2ff7, #00f5ff); box-shadow: 0 0 10px rgba(0,245,255,.4); }
.conf-fill-warn { background: linear-gradient(90deg, #ff8c00, #ff3250); box-shadow: 0 0 10px rgba(255,50,80,.4); }

/* ── MODULE CARDS ── */
.mod-card {
    background: rgba(123,47,247,0.07);
    border: 1px solid rgba(123,47,247,0.25);
    border-radius: 12px; padding: 18px 16px;
    margin-bottom: 12px;
}
.mod-title { font-family: 'Orbitron', monospace; font-size: 12px; font-weight: 700; color: #00f5ff; letter-spacing: 1px; margin-bottom: 4px; }
.mod-desc { font-size: 13px; color: rgba(180,180,220,.55); line-height: 1.5; }

/* ── THREAT CARDS ── */
.threat-card {
    background: rgba(0,245,255,.04);
    border: 1px solid rgba(0,245,255,.15);
    border-radius: 12px; padding: 18px;
    display: flex; align-items: flex-start; gap: 14px;
    margin-bottom: 12px;
}
.threat-dot { width:10px; height:10px; border-radius:50%; background:#00f5ff; box-shadow:0 0 10px #00f5ff; margin-top:5px; flex-shrink:0; }
.threat-title { font-weight:700; font-size:15px; color:#fff; margin-bottom:3px; }
.threat-desc { font-size:13px; color:rgba(180,180,220,.55); line-height:1.5; }

/* ── STREAMLIT OVERRIDES ── */
section[data-testid="stFileUploader"] > label { display: none !important; }
.stFileUploader > div { background: transparent !important; border: none !important; }

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(123,47,247,.2);
    border-radius: 12px; padding: 16px;
}
div[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    background: linear-gradient(90deg, #00f5ff, #7b2ff7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.stButton button {
    background: linear-gradient(135deg, #7b2ff7, #00c3ff) !important;
    color: #fff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important; font-size: 15px !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    border: none !important; border-radius: 8px !important;
    padding: 10px 28px !important;
    box-shadow: 0 0 25px rgba(123,47,247,.45) !important;
}
.stButton button:hover { box-shadow: 0 0 40px rgba(0,245,255,.6) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,.03) !important;
    border-radius: 10px;
    border: 1px solid rgba(123,47,247,.2);
    gap: 4px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 1px !important;
    color: rgba(200,200,255,.6) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(123,47,247,.3), rgba(0,245,255,.15)) !important;
    color: #00f5ff !important; border-radius: 8px !important;
}
hr { border-color: rgba(123,47,247,.2) !important; }

div[data-testid="stWarning"] {
    background: rgba(255,200,0,.08) !important;
    border: 1px solid rgba(255,200,0,.3) !important;
    border-radius: 10px !important;
    color: #ffc800 !important;
}
div[data-testid="stSpinner"] { color: #00f5ff !important; }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  NAVBAR
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="navbar">
    <div class="nav-logo">🛡 FORENSICS AI</div>
    <div class="nav-links">
        <a href="#">Detection</a>
        <a href="#">Analysis</a>
        <a href="#">Reports</a>
        <a href="#">Research</a>
        <span class="nav-btn">LIVE SYSTEM</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  HERO SECTION
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-section">
    <div class="hero-left">
        <div class="hero-badge">● REAL-TIME SYSTEM ONLINE</div>
        <div class="hero-title">Detect &amp; Expose</div>
        <span class="hero-title-accent">DeepFake Forensics</span>
        <p class="hero-sub">
            Real-time AI-powered image &amp; video authentication.
            Multimodal detection with explainable heatmaps,
            uncertainty scoring, and audio analysis —
            built to catch threats the others miss.
        </p>
    </div>
    <div class="hero-right">
        <div class="iso-frame">
            <div class="iso-rings">
                <div class="ring ring-1"></div>
                <div class="ring ring-2"></div>
                <div class="ring ring-3"></div>
            </div>
            <div class="orb orb-1"></div>
            <div class="orb orb-2"></div>
            <div class="orb orb-3"></div>
            <div class="iso-inner">
                <div class="iso-icon-main">🤖</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  STATS BAR
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-number">98.2%</span>
        <span class="stat-label">Detection Accuracy</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">&lt; 2s</span>
        <span class="stat-label">Real-Time Analysis</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">3-in-1</span>
        <span class="stat-label">Face · Audio · XAI</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">10+</span>
        <span class="stat-label">Deepfake Methods</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  SESSION STATE — persists results across Streamlit reruns
# ══════════════════════════════════════════════════════════
for key, default in {
    "results":     None,   # list of predict_face dicts
    "final_label": None,   # "FAKE" | "REAL"
    "avg_conf":    None,   # float
    "fake_count":  None,   # int
    "total":       None,   # int
    "media_type":  None,   # "image" | "video"
    "last_file":   None,   # filename — used to reset on new upload
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _render_result_card(final_label, avg_conf, fake_count, total, media_type):
    """Renders the styled result card. Shared by image + video paths."""
    label_class = "result-fake" if final_label == "FAKE" else "result-real"
    card_class  = "fake-card"   if final_label == "FAKE" else "real-card"
    icon        = "⚠️"         if final_label == "FAKE" else "✅"
    conf_pct    = int(avg_conf * 100)
    fill_class  = "conf-fill-warn" if final_label == "FAKE" else "conf-fill"

    row1_label = "Frames analyzed" if media_type == "video" else "Faces detected"
    row2_label = "Fake frames"     if media_type == "video" else "Flagged faces"

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-label {label_class}">{icon} {final_label}</div>
        <div class="result-meta">
            Confidence &nbsp;·&nbsp; <span>{avg_conf:.2%}</span><br>
            {row1_label} &nbsp;·&nbsp; <span>{total}</span><br>
            {row2_label} &nbsp;·&nbsp; <span>{fake_count}</span>
        </div>
        <div class="conf-wrap" style="margin-top:16px">
            <div class="conf-row">
                <span>Detection confidence</span><span>{conf_pct}%</span>
            </div>
            <div class="conf-track">
                <div class="{fill_class}" style="width:{conf_pct}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "  🎯  DETECTION ENGINE  ",
    "  📊  ANALYSIS REPORT  ",
    "  🔬  THREAT INTEL  "
])


# ══════════════════════════════════════════════════════════
#  TAB 1 — DETECTION ENGINE
# ══════════════════════════════════════════════════════════
with tab1:

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    # ── LEFT: UPLOAD ────────────────────────────────────
    with col_left:
        st.markdown("""
        <div class="section-heading">Upload Media</div>
        <p class="section-sub">Supports image (JPG, PNG) and video (MP4, MOV)</p>
        <div class="upload-zone">
            <div class="upload-icon">📁</div>
            <div class="upload-title">DROP FILE HERE</div>
            <p class="upload-hint">or use the browser below · max 200MB</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "mp4", "mov"])

        # reset session state when a new file is uploaded
        if uploaded and uploaded.name != st.session_state.last_file:
            for k in ("results", "final_label", "avg_conf", "fake_count", "total", "media_type"):
                st.session_state[k] = None
            st.session_state.last_file = uploaded.name

        if uploaded:
            st.markdown(f"""
            <div class="card" style="margin-top:14px">
                <div class="card-title">📎 FILE LOADED</div>
                <div class="card-desc">
                    {uploaded.name}<br>
                    {round(uploaded.size / 1024, 1)} KB · {uploaded.type}
                </div>
            </div>
            """, unsafe_allow_html=True)

            run = st.button("⚡  RUN FORENSIC ANALYSIS")
        else:
            run = False

    # ── RIGHT: MODULE INFO ───────────────────────────────
    with col_right:
        st.markdown("""
        <div class="section-heading">Active Modules</div>
        <p class="section-sub">All engines fire on upload</p>
        """, unsafe_allow_html=True)

        for icon, title, desc in [
            ("👁️", "FACIAL DEEPFAKE DETECTOR",
             "EfficientNet-B4 scans each detected face and flags manipulation regions"),
            ("📊", "CONFIDENCE SCORER",
             "Per-face confidence averaged across all detections for a final verdict"),
            ("🎥", "VIDEO FRAME PIPELINE",
             "Samples key frames, runs face analysis per frame, returns majority verdict"),
        ]:
            st.markdown(f"""
            <div class="mod-card">
                <div style="font-size:22px;margin-bottom:6px">{icon}</div>
                <div class="mod-title">{title}</div>
                <div class="mod-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════
    #  PROCESSING
    # ══════════════════════════════════════════════════════
    if uploaded and run:

        st.markdown("<hr>", unsafe_allow_html=True)
        file_type = uploaded.type

        # ── IMAGE PIPELINE ─────────────────────────────
        if "image" in file_type:

            try:
                uploaded.seek(0)   # reset stream — Streamlit may have already read it
                image = Image.open(uploaded).convert("RGB")
                image.load()       # force full decode before stream closes
            except Exception as e:
                st.error(f"❌ Could not open image: {e}")
                st.stop()

            img_col, res_col = st.columns([1, 1], gap="large")

            with img_col:
                st.markdown('<div class="section-heading" style="font-size:18px">Input Frame</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)

            with res_col:
                st.markdown('<div class="section-heading" style="font-size:18px">Forensic Result</div>', unsafe_allow_html=True)

                try:
                    with st.spinner("🔍 Running face detection…"):
                        faces = detect_faces(image)
                except Exception as e:
                    st.error(f"❌ Face detector failed: {e}")
                    st.stop()

                if len(faces) == 0:
                    st.warning("⚠️ No faces detected in this image.")
                    st.stop()

                try:
                    with st.spinner("🧠 Classifying faces…"):
                        results = [predict_face(f, threshold=0.70) for f in faces]
                except Exception as e:
                    st.error(f"❌ Classifier failed: {e}")
                    st.stop()

                # threshold-aware voting: uncertain predictions don't count
                confident   = [r for r in results if not r.get("uncertain", False)]
                voting      = confident if confident else results
                fake_count  = sum(r["label"] == "FAKE" for r in voting)
                final_label = "FAKE" if fake_count > len(voting) / 2 else "REAL"
                avg_conf    = sum(r["confidence"] for r in results) / len(results)
                any_uncertain = any(r.get("uncertain", False) for r in results)

                st.session_state.update({
                    "results": results, "final_label": final_label,
                    "avg_conf": avg_conf, "fake_count": fake_count,
                    "total": len(results), "media_type": "image",
                })

                _render_result_card(final_label, avg_conf, fake_count, len(results), "image")

                if any_uncertain:
                    st.markdown("""
                    <div style="background:rgba(255,200,0,.08);border:1px solid rgba(255,200,0,.3);
                    border-radius:10px;padding:10px 14px;margin-top:10px;font-size:13px;color:#ffc800">
                    ⚠️ One or more faces scored below the confidence threshold — result may be uncertain.
                    </div>""", unsafe_allow_html=True)

                if len(results) > 1:
                    st.markdown('<div class="card-title" style="margin-top:16px;color:#00f5ff">PER-FACE BREAKDOWN</div>', unsafe_allow_html=True)
                    for i, r in enumerate(results):
                        pct = int(r["confidence"] * 100)
                        lbl = r["label"]
                        unc = "  · uncertain" if r.get("uncertain") else ""
                        fc  = "conf-fill-warn" if lbl == "FAKE" else "conf-fill"
                        st.markdown(f"""
                        <div class="conf-wrap">
                            <div class="conf-row">
                                <span>Face {i+1} — {lbl}{unc}</span><span>{pct}%</span>
                            </div>
                            <div class="conf-track">
                                <div class="{fc}" style="width:{pct}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── VIDEO PIPELINE ─────────────────────────────
        elif "video" in file_type:

            st.markdown('<div class="section-heading" style="font-size:18px">Input Video</div>', unsafe_allow_html=True)
            st.video(uploaded)

            temp_path = "temp_video.mp4"
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded.read())
            except Exception as e:
                st.error(f"❌ Could not write temp file: {e}")
                st.stop()

            try:
                with st.spinner("🎥 Analyzing video frames…"):
                    results = analyze_video(temp_path)
            except Exception as e:
                st.error(f"❌ Video pipeline failed: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                st.stop()

            if os.path.exists(temp_path):
                os.remove(temp_path)

            if len(results) == 0:
                st.warning("⚠️ No frames could be analyzed.")
                st.stop()

            # threshold-aware majority voting
            confident   = [r for r in results if not r.get("uncertain", False)]
            voting      = confident if confident else results
            fake_count  = sum(r["label"] == "FAKE" for r in voting)
            final_label = "FAKE" if fake_count > len(voting) / 2 else "REAL"
            avg_conf    = sum(r["confidence"] for r in results) / len(results)
            uncertain_n = sum(1 for r in results if r.get("uncertain", False))

            st.session_state.update({
                "results": results, "final_label": final_label,
                "avg_conf": avg_conf, "fake_count": fake_count,
                "total": len(results), "media_type": "video",
            })

            v1, v2, v3, v4 = st.columns(4)
            with v1: st.metric("Verdict",           final_label)
            with v2: st.metric("Avg Confidence",    f"{avg_conf:.2%}")
            with v3: st.metric("Frames Analyzed",   len(results))
            with v4: st.metric("Uncertain Frames",  uncertain_n)

            st.markdown("<br>", unsafe_allow_html=True)
            _render_result_card(final_label, avg_conf, fake_count, len(voting), "video")

    # ── show persisted result if run already happened ──
    elif st.session_state.final_label is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading" style="font-size:18px">Last Result</div>', unsafe_allow_html=True)
        _render_result_card(
            st.session_state.final_label,
            st.session_state.avg_conf,
            st.session_state.fake_count,
            st.session_state.total,
            st.session_state.media_type,
        )


# ══════════════════════════════════════════════════════════
#  TAB 2 — ANALYSIS REPORT
# ══════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="section-heading">Analysis Report</div>
    <p class="section-sub">Forensic breakdown — run a detection first to populate live results</p>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Face Manipulation", "—", "awaiting input")
    with m2: st.metric("Avg Confidence",    "—", "awaiting input")
    with m3: st.metric("Frames / Faces",    "—", "awaiting input")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">📈 CONFIDENCE BREAKDOWN</div>
        <p class="card-desc" style="margin-top:10px">
            Upload and analyze a file in the Detection Engine tab to see
            per-face and per-frame confidence bars here.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  TAB 3 — THREAT INTEL
# ══════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div class="section-heading">Tackling Modern Threats</div>
    <p class="section-sub">Real-world deepfake threat categories this system targets</p>
    """, unsafe_allow_html=True)

    threats = [
        ("Voice Cloning Scams",
         "AI-generated voice replicas used in phone fraud, impersonating executives and family members."),
        ("Payment Verification Fraud",
         "Fake biometric images bypassing liveness checks in fintech and banking onboarding flows."),
        ("Non-Consensual Intimate Imagery",
         "Synthetic face-swaps placed on existing content — the fastest-growing real-world deepfake harm."),
        ("Video Call Impersonation",
         "Real-time face replacement in live video calls targeting corporate espionage and social engineering."),
    ]

    for title, desc in threats:
        st.markdown(f"""
        <div class="threat-card">
            <div class="threat-dot"></div>
            <div>
                <div class="threat-title">{title}</div>
                <div class="threat-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)