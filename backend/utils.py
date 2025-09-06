# backend/utils.py
from PIL import Image
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt

# --- Video ---
def extract_frames(file_path):
    # Return dummy frames for testing
    return [Image.new('RGB', (224,224), color='gray') for _ in range(5)]

def extract_audio_from_video(video_path, audio_path="temp_audio.wav"):
    # Dummy extraction (just create random audio)
    audio = np.random.rand(16000)
    # Save dummy audio if needed
    # librosa.output.write_wav(audio_path, audio, sr=16000)  # optional
    return audio

# --- Audio ---
def load_audio(file_path, sr=16000):
    # Dummy 1-second audio array
    return np.random.rand(sr)

def audio_to_melspec(audio, sr=16000):
    # Dummy Mel-spectrogram
    return np.abs(np.random.rand(128, 128))

# --- Heatmap ---
def hf_energy_map(frame):
    # Return frame itself for placeholder
    return frame

# --- Saving helpers ---
def save_image(image, path):
    image.save(path)

def save_spectrogram_image(melspec, path):
    plt.imshow(melspec, aspect='auto')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
