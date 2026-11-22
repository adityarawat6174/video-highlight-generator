"""Signal 3: speech excitement from Whisper transcript.
Hype-word density + punctuation emphasis + speech rate. Returns empty signal if Whisper
is unavailable (graceful degradation, Section 5.3 / 5.5)."""
import re
import numpy as np
from .utils import normalize

HYPE_WORDS = [
    "insane", "let's go", "lets go", "no way", "oh my god", "omg", "unbelievable",
    "crazy", "amazing", "wow", "clutch", "incredible", "what a", "yes", "boom",
    "goal", "win", "won", "champion", "important", "remember", "key point", "exam",
    "awesome", "fire", "epic", "holy",
]


def _segment_score(text, start, end):
    t = text.lower()
    words = re.findall(r"[\w']+", t)
    n = max(len(words), 1)
    hype = sum(t.count(h) for h in HYPE_WORDS) / n
    excl = text.count("!")
    caps = sum(1 for w in re.findall(r"[A-Za-z']+", text) if len(w) > 2 and w.isupper())
    rate = n / max(end - start, 0.5)
    return 3.0 * hype + 0.3 * excl + 0.3 * caps + 0.1 * rate


def speech_scores(wav_path, duration, step=1.0, model_name="base", log=print):
    """Return (times, scores, segments). Empty arrays on any failure."""
    try:
        import whisper  # openai-whisper
        model = whisper.load_model(model_name)
        result = model.transcribe(wav_path, fp16=False, verbose=False)
    except Exception as e:  # missing package, no internet, etc.
        log(f"[speech] Whisper unavailable ({type(e).__name__}: {str(e)[:80]}) -> skipping speech signal")
        return np.array([]), np.array([]), []
    segs = result.get("segments", [])
    times = np.arange(0, duration, step)
    scores = np.zeros_like(times)
    for s in segs:
        sc = _segment_score(s["text"], s["start"], s["end"])
        mask = (times + step > s["start"]) & (times < s["end"])
        scores[mask] = np.maximum(scores[mask], sc)
    return times, normalize(scores), [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in segs]
