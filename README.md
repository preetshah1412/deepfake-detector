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
