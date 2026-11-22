"""Signal 1: audio energy  S_audio = 0.6*RMS + 0.4*Onset  (Eq. 5.1, 5.2)."""
import os
import numpy as np
import librosa
from .utils import normalize, run, has_audio


def extract_audio(video_path, wav_path, sr=16000):
    run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr),
         "-acodec", "pcm_s16le", wav_path])
    return wav_path


def audio_scores(video_path, workdir, step=1.0, sr=16000):
    """Return (times, scores, wav_path). Scores are on a `step`-second grid."""
    if not has_audio(video_path):
        return np.array([]), np.array([]), None
    wav = extract_audio(video_path, os.path.join(workdir, "audio.wav"), sr)
    y, sr = librosa.load(wav, sr=sr, mono=True)
    hop = int(sr * 0.05)                      # 50 ms hop
    frame = hop * 4
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    n = min(len(rms), len(onset))
    rms, onset = normalize(rms[:n]), normalize(onset[:n])
    fused = 0.6 * rms + 0.4 * onset
    t_fine = np.arange(n) * hop / sr
    # aggregate (max) into `step`-second bins so short claps are not averaged away
    dur = len(y) / sr
    times = np.arange(0, dur, step)
    scores = np.array([fused[(t_fine >= t) & (t_fine < t + step)].max()
                       if np.any((t_fine >= t) & (t_fine < t + step)) else 0.0
                       for t in times])
    return times, normalize(scores), wav
