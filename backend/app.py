import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from .utils import extract_frames, extract_audio_from_video, load_audio, audio_to_melspec, hf_energy_map, save_image, save_spectrogram_image
from .video_model import VideoDeepfakeModel
from .audio_model import AudioDeepfakeModel
from .fusion import fuse_scores, trust_score

TMP_DIR = "tmp_outputs"
os.makedirs(TMP_DIR, exist_ok=True)

app = FastAPI(title="Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_model = VideoDeepfakeModel()
audio_model = AudioDeepfakeModel()

def analyze_video(path: str) -> Dict[str, Any]:
    frames = extract_frames(path, fps=5, max_frames=32)
    v_score = video_model.predict_frames(frames)

    # Save a few heatmaps for explainability
    heat_paths = []
    for i, fr in enumerate(frames[:6]):
        heat = hf_energy_map(fr)
        heat_path = save_image(heat, TMP_DIR, f"heat_{i}.png")
        heat_paths.append(heat_path)

    # Try audio from video as well
    try:
        y, sr = extract_audio_from_video(path, sr=16000)
        mel = audio_to_melspec(y, sr=sr)
        a_score = audio_model.predict_melspec(mel)
        mel_path = save_spectrogram_image(mel, TMP_DIR, "melspec.png")
    except Exception:
        a_score = None
        mel_path = None

    fused = fuse_scores(v_score, a_score, alpha=0.6)
    tscore = trust_score(fused)

    return {
        "type": "video",
        "video_fake_prob": v_score,
        "audio_fake_prob": a_score,
        "fused_fake_prob": fused,
        "trust_score": tscore,
        "heatmaps": heat_paths,
        "melspec_image": mel_path
    }

def analyze_audio(path: str) -> Dict[str, Any]:
    y, sr = load_audio(path, sr=16000)
    mel = audio_to_melspec(y, sr=sr)
    a_score = audio_model.predict_melspec(mel)
    mel_path = save_spectrogram_image(mel, TMP_DIR, "melspec.png")
    tscore = trust_score(a_score)
    return {
        "type": "audio",
        "audio_fake_prob": a_score,
        "trust_score": tscore,
        "melspec_image": mel_path
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = (file.filename or "").lower()
    tmp_path = os.path.join(TMP_DIR, file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if any(suffix.endswith(ext) for ext in [".mp4", ".mov", ".mkv", ".avi"]):
        out = analyze_video(tmp_path)
    elif any(suffix.endswith(ext) for ext in [".wav", ".mp3", ".flac", ".m4a"]):
        out = analyze_audio(tmp_path)
    else:
        out = {"error": "Unsupported file type. Please upload video (.mp4) or audio (.wav/.mp3)."}
    return out
