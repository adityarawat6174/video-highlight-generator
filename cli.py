"""Command line:  python cli.py my_video.mp4 --k 5 --out output --no-whisper"""
import argparse
from highlightgen import run_pipeline

ap = argparse.ArgumentParser(description="AI-Based Video Highlight Generator")
ap.add_argument("video")
ap.add_argument("--out", default="output")
ap.add_argument("--k", type=int, default=5, help="number of highlights")
ap.add_argument("--no-whisper", action="store_true")
ap.add_argument("--whisper-model", default="base")
ap.add_argument("--no-reel", action="store_true")
a = ap.parse_args()
rep, _ = run_pipeline(a.video, a.out, a.k, use_whisper=not a.no_whisper,
                      whisper_model=a.whisper_model, build_reel_flag=not a.no_reel)
for h in rep["highlights"]:
    print(f"#{h['rank']}  {h['start']:.1f}s - {h['end']:.1f}s  score={h['score']}")
print("Signals used:", rep["signals_used"], "| report:", a.out + "/report.json")
