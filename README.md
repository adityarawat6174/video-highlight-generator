# AI-Based Video Highlight Generator

Fuses **audio energy**, **Whisper speech transcript** and **visual scene changes** into one score
curve, picks the top-k peaks, pads/merges them and cuts clips with FFmpeg.

## Setup
1. Install FFmpeg (`sudo apt install ffmpeg` / `brew install ffmpeg` / `choco install ffmpeg`)
2. `pip install -r requirements.txt`
3. (Optional, for the speech signal) `pip install openai-whisper`

## Run the web UI
    streamlit run app.py
Click **"Use synthetic demo video"** for an instant demo, or upload your own video.

## Run from command line
    python make_test_video.py test_video.mp4
    python cli.py test_video.mp4 --k 3 --no-whisper
    python cli.py my_stream.mp4 --k 5            # with Whisper

## Structure
    highlightgen/audio_module.py   RMS + onset strength          (Eq 5.1, 5.2)
    highlightgen/visual_module.py  histogram diff + optical flow (Eq 5.3, 5.4)
    highlightgen/speech_module.py  Whisper + hype words, auto-skips if unavailable
    highlightgen/fusion.py         weighted fusion + Algorithm 1  (Eq 5.5)
    highlightgen/clipper.py        FFmpeg clips + reel
    highlightgen/pipeline.py       orchestration + report.json
    app.py / cli.py                Streamlit UI / command line
    make_test_video.py             synthetic ground-truth video (Sec 5.9)
