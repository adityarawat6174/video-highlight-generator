"""Clip extraction and highlight reel with FFmpeg (libx264 + aac)."""
import os
from .utils import run


def extract_clips(video_path, segments, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for s in segments:
        path = os.path.join(out_dir, f"highlight_{s['rank']:02d}_{s['start']:.0f}s-{s['end']:.0f}s.mp4")
        run(["ffmpeg", "-y", "-ss", str(s["start"]), "-i", video_path,
             "-t", str(round(s["end"] - s["start"], 2)),
             "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-movflags", "+faststart", path])
        s["clip_path"] = path
    return segments


def build_reel(segments, out_path):
    """Concatenate clips in chronological order into one reel."""
    ordered = sorted(segments, key=lambda s: s["start"])
    list_file = out_path + ".txt"
    with open(list_file, "w") as f:
        for s in ordered:
            f.write(f"file '{os.path.abspath(s['clip_path'])}'\n")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", out_path])
    os.remove(list_file)
    return out_path
