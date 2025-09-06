# Deepfake Video & Voice Detector (Python-first, 24h Hackathon Starter)

A lightweight, privacy-first deepfake detection prototype that analyzes both **video frames** and **audio spectrograms** to produce a **Trust Score** with simple explainability overlays. Built for fast demos with **Streamlit** UI and an optional **FastAPI** backend.

## Features
- Upload **video** (`.mp4`, `.mov`, `.mkv`) or **audio** (`.wav`, `.mp3`).
- Extract frames (video) and mel-spectrogram (audio).
- Run lightweight CNNs (placeholders) with optional `models/*.pth` weights.
- Display **Trust Score** and **heatmaps** for suspicious regions.
- Works fully **offline** (no external calls).

## Quickstart (Local, one terminal)
```bash
# 1) Create venv and install
python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) Launch Streamlit UI (uses local modules directly)
streamlit run frontend/streamlit_app.py
```

Open the printed local URL in your browser, upload a clip, and test.

## Optional: Run FastAPI backend + Streamlit UI (two terminals)
Terminal A:
```bash
uvicorn backend.app:app --reload --port 8000
```
Terminal B:
```bash
BACKEND_URL=http://127.0.0.1:8000 streamlit run frontend/streamlit_app.py
```

## Models
Place your fine-tuned weights here (optional):
```
models/
├── video_model.pth   # state_dict for VideoDeepfakeModel
└── audio_model.pth   # state_dict for AudioDeepfakeModel
```
If weights are missing, the app falls back to heuristic+ImageNet features (good enough for demo flow).

## Docker (single image running Streamlit only for simplicity)
```bash
docker build -t deepfake-detector .
docker run -p 8501:8501 deepfake-detector
```
Then open http://localhost:8501

## Notes
- This repo is **starter code** optimized for a 24‑hour hackathon. For production, use robust, vetted models (e.g., trained on FaceForensics++, ASVspoof) and stronger explainability (Grad-CAM on the last conv layer).
- The placeholder heatmap uses high-frequency energy to highlight likely manipulation artifacts—useful for demo visuals.
