"""Streamlit UI:  streamlit run app.py"""
import os, json, tempfile, shutil
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from highlightgen import run_pipeline

st.set_page_config(page_title="AI Video Highlight Generator", page_icon="🎬", layout="wide")
st.title("🎬 AI-Based Video Highlight Generator")
st.caption("Audio energy + Speech (Whisper) + Visual changes → weighted fusion → ranked highlight clips")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of highlights", 1, 10, 5)
    pre = st.slider("Pre-padding (s)", 0, 10, 3)
    post = st.slider("Post-padding (s)", 0, 15, 5)
    gap = st.slider("Minimum gap between peaks (s)", 2, 60, 8)
    use_whisper = st.checkbox("Use Whisper speech signal", value=False,
                              help="Needs `pip install openai-whisper` and internet for the first model download. "
                                   "If unavailable, the system auto re-weights the other signals.")
    wmodel = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1, disabled=not use_whisper)
    reel = st.checkbox("Build combined highlight reel", value=True)
    st.subheader("Fusion weights")
    wa = st.slider("Audio", 0.0, 1.0, 0.40, 0.05)
    wt = st.slider("Speech", 0.0, 1.0, 0.35, 0.05, disabled=not use_whisper)
    wv = st.slider("Visual", 0.0, 1.0, 0.25, 0.05)
    st.caption("Weights are re-normalised to sum to 1 over the signals that are actually available.")

# ---------------- input ----------------
c1, c2 = st.columns([3, 1])
with c1:
    up = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi", "webm"])
with c2:
    st.write("")
    st.write("")
    demo = st.button("Use synthetic demo video")

if demo:
    import subprocess, sys
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "synthetic_test.mp4")
    subprocess.run([sys.executable, "make_test_video.py", p], check=True, capture_output=True)
    st.session_state["video_path"] = p
    st.session_state["is_demo"] = True
elif up is not None:
    if st.session_state.get("uploaded_name") != up.name:
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, up.name)
        with open(p, "wb") as f:
            f.write(up.getbuffer())
        st.session_state.update(video_path=p, uploaded_name=up.name, is_demo=False)

vp = st.session_state.get("video_path")
if vp:
    st.video(vp)
    if st.button("🚀 Generate highlights", type="primary"):
        out = tempfile.mkdtemp(prefix="hl_")
        bar = st.progress(0.0, text="Starting...")
        try:
            rep, curves = run_pipeline(
                vp, out, k=k, use_whisper=use_whisper, whisper_model=wmodel,
                build_reel_flag=reel, weights={"audio": wa, "speech": wt, "visual": wv},
                pre_pad=pre, post_pad=post, min_gap=gap,
                progress=lambda f, m: bar.progress(f, text=m), log=lambda m: None)
            st.session_state["result"] = (rep, curves)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
        bar.empty()

# ---------------- results ----------------
if "result" in st.session_state:
    rep, curves = st.session_state["result"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Video length", f"{rep['duration_sec']:.0f}s")
    m2.metric("Highlights found", len(rep["highlights"]))
    m3.metric("Signals used", ", ".join(rep["signals_used"]))
    m4.metric("Processing time", f"{rep['processing_time_sec']}s")
    if "speech" not in rep["signals_used"]:
        st.info("Speech signal not used → weights automatically redistributed over: "
                + ", ".join(f"{k} {v:.2f}" for k, v in rep["effective_weights"].items()))

    tab1, tab2, tab3 = st.tabs(["🎞️ Highlights", "📈 Score curves", "🧾 Report (JSON)"])
    with tab1:
        if rep.get("reel") and os.path.exists(rep["reel"]):
            st.subheader("Combined highlight reel")
            st.video(rep["reel"])
            with open(rep["reel"], "rb") as f:
                st.download_button("⬇️ Download reel", f, "highlight_reel.mp4", "video/mp4")
        st.subheader("Individual clips (ranked by score)")
        cols = st.columns(2)
        for i, h in enumerate(rep["highlights"]):
            with cols[i % 2]:
                st.markdown(f"**#{h['rank']}** · {h['start']:.0f}s – {h['end']:.0f}s · "
                            f"peak @ {h['peak_time']:.0f}s · score **{h['score']:.2f}**")
                st.video(h["clip_path"])
                if h.get("transcript"):
                    st.caption(f"🗣️ {h['transcript']}")
                with open(h["clip_path"], "rb") as f:
                    st.download_button(f"⬇️ Download clip #{h['rank']}", f,
                                       os.path.basename(h["clip_path"]), "video/mp4", key=f"dl{i}")
    with tab2:
        fig, ax = plt.subplots(figsize=(11, 3.8))
        t = curves["time"]
        for name, color in [("audio", "tab:blue"), ("visual", "tab:green"), ("speech", "tab:purple")]:
            if name in curves:
                ax.plot(t, curves[name], label=f"{name} score", color=color, alpha=0.6, lw=1)
        ax.plot(t, curves["fused"], label="fused score", color="red", lw=2.2)
        for h in rep["highlights"]:
            ax.axvspan(h["start"], h["end"], color="gold", alpha=0.3)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalised score (0-1)")
        ax.set_title("Per-signal and fused highlight scores (gold = selected highlights)")
        ax.legend(loc="upper right"); ax.grid(alpha=0.3)
        st.pyplot(fig)
    with tab3:
        st.json({k_: v for k_, v in rep.items()})
        st.download_button("⬇️ Download report.json", json.dumps(rep, indent=2),
                           "report.json", "application/json")
