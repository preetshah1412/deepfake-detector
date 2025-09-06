# backend/utils.py
from PIL import Image
import numpy as np

def extract_frames(file_path):
    # Return dummy frames for testing
    return [Image.new('RGB', (224,224), color='gray') for _ in range(5)]

def extract_audio(file_path):
    # Return dummy 1-second audio array for testing
    return np.random.rand(16000)

def generate_heatmap(frame):
    # Just return the frame itself as placeholder
    return frame
