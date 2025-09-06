# frontend/streamlit_app.py

import sys, os
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import time

# --- Add project root to Python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Fallback backend for instant demo ---
try:
    from backend.video_model import VideoDeepfakeModel
    from backend.audio_model import AudioDeepfakeModel
    from backend.fusion import fuse_scores
    from backend.utils import extract_frames, extract_audio, generate_heatmap
except:
    class VideoDeepfakeModel:
        def __init__(self, path): pass
        def predict(self, frame): return 90
    class AudioDeepfakeModel:
        def __init__(self, path): pass
        def predict(self, audio): return 85
    def fuse_scores(v,a): return 0.6*v + 0.4*a
    def extract_frames(file_path): return [Image.new('RGB',(224,224),'gray') for _ in range(5)]
    def extract_audio(file_path): return np.random.rand(16000)
    def generate_heatmap(frame): return frame

# --- Page config ---
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for premium mobile UI ---
st.markdown("""
<style>
body {background: linear-gradient(135deg,#0f111a,#1a1c2b); color:#fff; font-family:'Inter', sans-serif;}
.card {background-color:#1e2033; padding:20px; border-radius:25px; box-shadow:0 10px 25px rgba(0,0,0,0.3); margin-bottom:20px;}
.stButton>button {background:linear-gradient(90deg,#4f46e5,#6366f1); color:white; border-radius:20px; padding:12px 30px; font-size:18px; border:none; transition:0.3s;}
.stButton>button:hover {opacity:0.9; transform:scale(1.05);}
.trust-score {font-size:60px; font-weight:bold; text-align:center; margin:20px 0;}
.scroll-row { display:flex; overflow-x:auto; gap:15px; padding-bottom:10px; }
.scroll-row img { border-radius:20px; box-shadow:0 5px 15px rgba(0,0,0,0.3); height:150px; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("## 🛡️ Deepfake Video & Voice Detector", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#facc15;'>Verify videos & audio instantly!</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Upload Section ---
st.markdown('<div class="card"><h3>Upload Video or Audio</h3></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a video or audio file", type=["mp4","mov","avi","wav","mp3"])

if uploaded_file is not None:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # --- Analysis Spinner ---
    with st.spinner("Analyzing..."):
        frames = extract_frames(file_path)
        audio_data = extract_audio(file_path)

        # Dummy ML for fast demo
        video_score = VideoDeepfakeModel("").predict(frames[0])
        audio_score = AudioDeepfakeModel("").predict(audio_data)
        trust_score = fuse_scores(video_score, audio_score)

        time.sleep(0.5)  # Small delay for animation effect

    # --- Animated Trust Score ---
    score_placeholder = st.empty()
    for i in range(0, int(trust_score)+1, 2):
        color = "#10b981" if i>70 else "#facc15" if i>40 else "#ef4444"
        score_placeholder.markdown(f'<div class="trust-score" style="color:{color}">{i}% Trust</div>', unsafe_allow_html=True)
        time.sleep(0.02)
    score_placeholder.markdown(f'<div class="trust-score" style="color:{color}">{trust_score:.1f}% Trust</div>', unsafe_allow_html=True)

    # --- Video Heatmaps Carousel ---
    st.markdown('<div class="card"><h4>Video Frame Anomalies</h4></div>', unsafe_allow_html=True)
    st.markdown('<div class="scroll-row">', unsafe_allow_html=True)
    for frame in frames[:10]:  # first 10 frames
        heatmap = generate_heatmap(frame)
        st.image(heatmap, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Audio Spectrogram ---
    st.markdown('<div class="card"><h4>Audio Spectrogram</h4></div>', unsafe_allow_html=True)
    S = librosa.feature.melspectrogram(y=audio_data, sr=16000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(6,2))
    librosa.display.specshow(S_dB, sr=16000, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title="Mel Spectrogram")
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa;'>Hackathon Demo – NFT/crypto mobile-style UI 🚀</p>", unsafe_allow_html=True)
