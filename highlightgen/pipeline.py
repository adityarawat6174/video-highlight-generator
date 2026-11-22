"""Orchestrates the whole system and writes report.json (transparency, Section 4.5)."""
import json
import os
import time
import numpy as np
from .utils import video_duration
from .audio_module import audio_scores
from .visual_module import visual_scores
from .speech_module import speech_scores
from .fusion import fuse, select_segments, DEFAULT_WEIGHTS
from .clipper import extract_clips, build_reel


def run_pipeline(video_path, out_dir="output", k=5, step=1.0, use_whisper=True,
                 whisper_model="base", build_reel_flag=True, weights=None,
                 pre_pad=3.0, post_pad=5.0, min_gap=8.0, progress=None, log=print):
    """progress(fraction, message) is optional (used by the Streamlit UI)."""
    def P(f, m):
        log(f"[{int(f*100):3d}%] {m}")
        if progress:
            progress(f, m)

    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    duration = video_duration(video_path)

    P(0.05, "Analysing audio energy (RMS + onset)...")
    a_t, a_s, wav = audio_scores(video_path, out_dir, step)

    P(0.25, "Analysing visual changes (histogram + optical flow)...")
    v_t, v_s = visual_scores(video_path, step)

    s_t, s_s, transcript = np.array([]), np.array([]), []
    if use_whisper and wav:
        P(0.6, "Transcribing speech with Whisper...")
        s_t, s_s, transcript = speech_scores(wav, duration, step, whisper_model, log)
    else:
        log("[speech] disabled -> using remaining signals")

    P(0.75, "Fusing signals and detecting peaks...")
    grid, fused, per_signal, used_w = fuse(
        {"audio": (a_t, a_s), "visual": (v_t, v_s), "speech": (s_t, s_s)},
        duration, step, weights)
    segments = select_segments(grid, fused, k, min_gap, pre_pad, post_pad, duration)

    P(0.85, "Cutting clips with FFmpeg...")
    extract_clips(video_path, segments, os.path.join(out_dir, "clips"))
    reel = None
    if build_reel_flag and segments:
        reel = build_reel(segments, os.path.join(out_dir, "highlight_reel.mp4"))

    # attach transcript text to each highlight (nice for demo)
    for s in segments:
        s["transcript"] = " ".join(t["text"] for t in transcript
                                   if t["end"] > s["start"] and t["start"] < s["end"])

    report = {
        "video": os.path.basename(video_path),
        "duration_sec": round(duration, 2),
        "signals_used": list(used_w.keys()),
        "effective_weights": {k_: round(v, 3) for k_, v in used_w.items()},
        "default_weights": DEFAULT_WEIGHTS,
        "highlights": segments,
        "reel": reel,
        "processing_time_sec": round(time.time() - t0, 2),
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    curves = {"time": grid, "fused": fused, **per_signal}
    P(1.0, "Done.")
    return report, curves
