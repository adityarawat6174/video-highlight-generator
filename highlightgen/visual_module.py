"""Signal 2: visual excitement = 0.5*histogram-difference + 0.5*Farneback flow (Eq. 5.3, 5.4)."""
import cv2
import numpy as np
from .utils import normalize


def visual_scores(video_path, step=1.0, analysis_width=320, progress=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if total > 0 else 0
    times, hist_s, flow_s = [], [], []
    prev_gray, prev_hist = None, None
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (analysis_width, int(h * analysis_width / w)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if prev_gray is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            hist_s.append(1.0 - corr)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_s.append(float(np.mean(np.linalg.norm(flow, axis=2))))
        else:
            hist_s.append(0.0)
            flow_s.append(0.0)
        times.append(t)
        prev_gray, prev_hist = gray, hist
        t += step
        if progress:
            progress(min(t / duration, 1.0))
    cap.release()
    if not times:
        return np.array([]), np.array([])
    scores = 0.5 * normalize(hist_s) + 0.5 * normalize(flow_s)
    return np.array(times), normalize(scores)
