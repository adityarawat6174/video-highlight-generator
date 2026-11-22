"""Weighted fusion (Eq. 5.5) and Algorithm 1 (peak selection, padding, merging)."""
import numpy as np
from .utils import normalize

DEFAULT_WEIGHTS = {"audio": 0.40, "speech": 0.35, "visual": 0.25}


def fuse(signals, duration, step=1.0, weights=None):
    """signals: {name: (times, scores)}. Unavailable (empty) signals are dropped and the
    remaining weights are re-normalised so they always sum to 1."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    grid = np.arange(0, duration, step)
    active = {k: v for k, v in signals.items() if len(v[0]) > 1 and k in weights}
    if not active:
        raise RuntimeError("No usable signals - cannot compute highlights.")
    total = sum(weights[k] for k in active)
    used = {k: weights[k] / total for k in active}
    fused = np.zeros_like(grid)
    resampled = {}
    for k, (t, s) in active.items():
        r = np.interp(grid, t, s)
        resampled[k] = r
        fused += used[k] * r
    return grid, normalize(fused), resampled, used


def select_segments(grid, fused, k=5, min_gap=8.0, pre_pad=3.0, post_pad=5.0, duration=None):
    """Algorithm 1: greedy top-k peaks with min separation -> padded -> merged -> ranked."""
    duration = duration or grid[-1]
    chosen = []
    for i in np.argsort(-fused):
        t = grid[i]
        if all(abs(t - c) >= min_gap for c, _ in chosen):
            chosen.append((t, float(fused[i])))
        if len(chosen) == k:
            break
    segs = sorted([[max(0.0, t - pre_pad), min(duration, t + post_pad), sc, t]
                   for t, sc in chosen], key=lambda s: s[0])
    merged = []
    for s in segs:
        if merged and s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
            if s[2] > merged[-1][2]:
                merged[-1][2], merged[-1][3] = s[2], s[3]
        else:
            merged.append(s)
    merged.sort(key=lambda s: -s[2])
    return [{"rank": i + 1, "start": round(s[0], 2), "end": round(s[1], 2),
             "peak_time": round(s[3], 2), "score": round(s[2], 4)}
            for i, s in enumerate(merged)]
