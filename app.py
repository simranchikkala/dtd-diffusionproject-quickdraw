"""
PLACEHOLDER APP - to be replaced by the real app.

This minimal version exists so the Docker image and deployment can be built and
tested before the real app is ready. It loads model.pt, takes an uploaded
doodle (or a built-in example star), shrinks it to 8x8, adds noise, and shows
the diffusion model's 28x28 reconstruction.

Run locally:   streamlit run app.py
"""

import os

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

from src.data import make_hint, preprocess_image, to_display
from src.diffusion import Diffusion
from src.metrics import mse, ssim
from src.models import UNet, load_checkpoint

MODEL_PATH = "model.pt"

st.set_page_config(page_title="Doodle Diffusion", page_icon="⭐")


@st.cache_resource
def load_model():
    model, ckpt = load_checkpoint(MODEL_PATH, UNet)
    return model, Diffusion(ckpt.get("timesteps", 200))


def example_star():
    """A simple black-on-white star, used when nothing is uploaded."""
    img = Image.new("RGB", (280, 280), "white")
    cx, cy, r_out, r_in = 140, 145, 115, 48
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        a = np.pi / 2 + i * np.pi / 5
        pts.append((cx + r * np.cos(a), cy - r * np.sin(a)))
    ImageDraw.Draw(img).line(pts + [pts[0]], fill="black", width=24, joint="curve")
    return img


st.title("Doodle Diffusion")
st.caption("Placeholder app for testing Docker and deployment. The full app is coming soon.")

if not os.path.exists(MODEL_PATH):
    st.error("model.pt not found. Train it first with: python train.py")
    st.stop()

with st.sidebar:
    noise = st.slider("Noise added to the 8x8 input", 0.0, 0.3, 0.1, 0.05)
    steps = st.slider("Diffusion steps", 10, 100, 50, 10)
    seed = st.number_input("Random seed", 0, 9999, 42)

upload = st.file_uploader("Upload a star or moon doodle (optional)",
                          type=["png", "jpg", "jpeg"])
image = Image.open(upload) if upload else example_star()

x, blank = preprocess_image(image)
if blank:
    st.warning("The image looks blank. Try a drawing with clear dark lines.")
    st.stop()

model, diffusion = load_model()
hint = make_hint(x, noise, torch.Generator().manual_seed(int(seed)))
with st.spinner("Generating..."):
    out = diffusion.sample(model, hint, steps=steps, seed=int(seed))

cols = st.columns(3)
cols[0].image(to_display(x), caption="Your doodle (28x28)")
cols[1].image(to_display(hint.clamp(0, 1)), caption="Noisy 8x8 input")
cols[2].image(to_display(out), caption="Diffusion output (28x28)")
st.write(f"MSE: {mse(out, x).item():.4f}   |   SSIM: {ssim(out, x).item():.3f}")