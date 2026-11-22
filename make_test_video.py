"""Synthetic ground-truth video (Section 5.9): 64 s, planted highlights at 12-17, 31-36, 49-54 s."""
import subprocess, sys

out = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
windows = [(12, 17), (31, 36), (49, 54)]
total = 64
segs, cur = [], 0
for a, b in windows:
    segs += [("calm", cur, a), ("exc", a, b)]
    cur = b
segs.append(("calm", cur, total))

inputs, vf, af = [], [], []
for i, (kind, a, b) in enumerate(segs):
    d = b - a
    if kind == "calm":
        v = f"testsrc=size=320x240:rate=25:duration={d}"
        au = f"sine=frequency=440:sample_rate=16000:duration={d},volume=0.05"
    else:
        v = f"mandelbrot=size=320x240:rate=25,trim=duration={d}"
        au = f"sine=frequency=1000:sample_rate=16000:duration={d},volume=0.9,tremolo=f=8:d=0.9"
    inputs += ["-f", "lavfi", "-i", v, "-f", "lavfi", "-i", au]
n = len(segs)
fc = "".join(f"[{2*i}:v][{2*i+1}:a]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"
subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out],
               check=True, capture_output=True)
print("created", out)
