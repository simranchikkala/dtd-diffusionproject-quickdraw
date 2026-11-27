"""
Quick, Draw! Diffusion - doodle super-resolution web app.

The user draws (or uploads, or picks) a star or moon doodle. The app:
  1. turns it into a Quick, Draw!-style 28x28 image,
  2. shrinks it to 8x8 and adds noise (the brief's formula), which is all the
     model gets to see,
  3. runs the conditional diffusion model to generate a sharp 28x28 doodle,
  4. compares it with the bicubic, linear regression and CNN baselines, and
     checks every output with the separately trained star/moon classifier.

Run locally:   python -m streamlit run app.py
"""

import base64
import io
import os

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image, ImageDraw

from src.data import downsample, make_hint, normalise, preprocess_image, to_display, upsample
from src.diffusion import Diffusion
from src.metrics import mse, ssim
from src.models import CNNSR, LinearSR, ShapeClassifier, UNet, load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pt")
MODELS_DIR = os.path.join(HERE, "models")
CANVAS_SIZE = 280

st.set_page_config(page_title="Doodle Diffusion", page_icon="⭐", layout="wide")

draw_canvas = components.declare_component(
    "draw_canvas", path=os.path.join(HERE, "components", "draw_canvas"))


# ----------------------------------------------------------------------------- models
@st.cache_resource
def load_models():
    """Load every trained model once and share it between all users."""
    unet, ckpt = load_checkpoint(MODEL_PATH, UNet)
    models = {"unet": unet, "diffusion": Diffusion(ckpt.get("timesteps", 200)),
              "categories": ckpt.get("categories", ["star", "moon"])}
    for name, cls in [("linear", LinearSR), ("cnn", CNNSR), ("classifier", ShapeClassifier)]:
        path = os.path.join(MODELS_DIR, f"{name}.pt")
        models[name] = load_checkpoint(path, cls)[0] if os.path.exists(path) else None
    return models


@torch.no_grad()
def classify(models, images):
    """Return (B, n_categories) probabilities from the star/moon classifier, or None."""
    if models["classifier"] is None:
        return None
    return torch.softmax(models["classifier"](images), dim=1)


@st.cache_data(show_spinner=False, max_entries=50)
def run_pipeline(x_bytes, noise, steps, guidance, seed, n_samples):
    """Everything the page shows, computed once per input + settings.

    x_bytes is the 28x28 doodle as raw float32 bytes, so Streamlit can cache on it.
    """
    models = load_models()
    unet, diffusion = models["unet"], models["diffusion"]
    x = torch.frombuffer(bytearray(x_bytes), dtype=torch.float32).view(1, 1, 28, 28)

    shrunk = normalise(downsample(x))
    hint = make_hint(x, noise, torch.Generator().manual_seed(seed))

    out, frames = diffusion.sample(unet, hint, steps=steps, guidance=guidance,
                                   seed=seed, n_frames=8)
    outputs = {"Bicubic": upsample(hint.clamp(0, 1))}
    with torch.no_grad():
        if models["linear"] is not None:
            outputs["Linear regression"] = models["linear"](hint).clamp(0, 1)
        if models["cnn"] is not None:
            outputs["CNN"] = models["cnn"](hint).clamp(0, 1)
    outputs["Diffusion"] = out

    # Several samples from the same hint: one batch, one generator -> different noise each.
    samples = diffusion.sample(unet, hint.repeat(n_samples, 1, 1, 1), steps=steps,
                               guidance=guidance, seed=seed + 1)

    stacked = torch.cat(list(outputs.values()))
    return {
        "x": x, "shrunk": shrunk, "hint": hint, "frames": frames,
        "outputs": outputs, "samples": samples,
        "mse": mse(stacked, x.expand_as(stacked)).tolist(),
        "ssim": ssim(stacked, x.expand_as(stacked)).tolist(),
        "probs_input": classify(models, x),
        "probs_outputs": classify(models, stacked),
        "probs_samples": classify(models, samples),
        "categories": models["categories"],
    }


# ----------------------------------------------------------------------------- inputs
def example_star():
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    cx, cy, r_out, r_in = 140, 148, 115, 46
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        a = np.pi / 2 + i * np.pi / 5
        pts.append((cx + r * np.cos(a), cy - r * np.sin(a)))
    ImageDraw.Draw(img).line(pts + [pts[0]], fill="black", width=16, joint="curve")
    return img


def example_moon():
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    d = ImageDraw.Draw(img)
    d.arc((30, 30, 250, 250), start=60, end=300, fill="black", width=16)   # outer curve
    d.arc((95, 45, 245, 235), start=95, end=265, fill="black", width=16)   # inner curve
    return img


def example_star_5_point():
    """A star drawn the way most people doodle it: five crossing lines in one stroke."""
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    cx, cy, r = 140, 150, 120
    outer = [(cx + r * np.cos(np.pi / 2 + i * 2 * np.pi / 5),
              cy - r * np.sin(np.pi / 2 + i * 2 * np.pi / 5)) for i in range(5)]
    order = [outer[i] for i in (0, 2, 4, 1, 3, 0)]
    ImageDraw.Draw(img).line(order, fill="black", width=16, joint="curve")
    return img


EXAMPLES = {
    "Star (outline)": example_star,
    "Star (one stroke)": example_star_5_point,
    "Moon": example_moon,
}


def decode_data_url(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


# ----------------------------------------------------------------------------- display helpers
def show(container, x, caption, size=180):
    container.image(to_display(x.clamp(0, 1), size=size), caption=caption)


def verdict(probs, categories, i=0):
    if probs is None:
        return ""
    p = probs[i]
    k = int(p.argmax())
    return f"Looks like a **{categories[k]}** ({p[k]:.0%})"


def frames_strip(frames, size=112, gap=6):
    """All denoising frames side by side in one image."""
    strip = Image.new("L", (len(frames) * (size + gap) - gap, size), 255)
    for i, f in enumerate(frames):
        strip.paste(to_display(f, size=size), (i * (size + gap), 0))
    return strip


def frames_gif(frames, size=196):
    """Turn the denoising frames into an animated GIF (holding on the final image)."""
    imgs = [to_display(f, size=size).convert("P") for f in frames]
    imgs += [imgs[-1]] * 4
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=350, loop=0)
    return buf.getvalue()


def render_about():
    st.header("How it works")
    st.markdown("""
1. **Shrink.** A 28×28 doodle is reduced to **8×8**, losing most of its detail. Each 8×8
   image is brightened so its strongest pixel is 1, so faint strokes survive.
2. **Add noise.** Noise is added with the brief's formula
   **z = √a · x + √(1 − a) · ε**, where the noise level is 1 − a.
3. **Generate.** A conditional **U-Net** diffusion model starts from pure noise at 28×28 and
   removes it step by step, using the noisy 8×8 image as a guide, until a doodle appears.
   We use DDIM sampling so 50 steps are enough.

We compared four methods, following the brief's *Baseline → Evaluate → Identify Limitation →
Improve Model* approach:

| Method | Idea |
|---|---|
| Bicubic upscaling | Stretches the image; no learning |
| Linear regression | Each output pixel is a weighted sum of the 64 input pixels (the brief's baseline) |
| CNN | Small convolutional network that uses local spatial structure |
| Conditional diffusion (U-Net) | Generates the image from noise, guided by the 8×8 input |
""")

    st.header("Results")
    st.markdown("300 held-out test drawings (146 stars, 154 moons), noise level 0.1, "
                "guidance 1, 50 sampling steps.")
    st.markdown("""
| Method | MSE ↓ | SSIM ↑ | Shape accuracy ↑ |
|---|---|---|---|
| Bicubic | 0.117 | 0.208 | 0.51 |
| Linear regression | 0.064 | 0.212 | 0.51 |
| CNN | 0.066 | 0.221 | 0.52 |
| Diffusion | 0.128 | 0.170 | **0.77** |
| Real drawings (ceiling) | 0 | 1 | 0.98 |
""")
    st.markdown("""
- **MSE** (average pixel error) rewards safe, blurry outputs, because a blur "hedges" between
  possible stroke positions.
- **SSIM** (structural similarity) also compares pixel by pixel, so a sharp stroke drawn
  slightly out of place is penalised heavily.
- **Shape accuracy** uses a separately trained star/moon classifier to check whether each
  output is recognisably the right shape. Its accuracy on real drawings (0.98) is the ceiling.

The baselines score best on MSE and SSIM, but their blurry outputs all look like moons: they
recognise 0–1% of stars. **Diffusion draws the right shape 77% of the time** (82% of stars,
71% of moons), at the cost of higher pixel error.
""")

    st.subheader("Effect of guidance strength")
    st.markdown("""
| Guidance | MSE ↓ | SSIM ↑ | Shape accuracy ↑ |
|---|---|---|---|
| 1 (used) | 0.128 | 0.170 | 0.77 |
| 1.5 | 0.157 | 0.135 | 0.63 |
| 2 | 0.207 | 0.087 | 0.49 |
| 3 | 0.281 | 0.033 | 0.50 |
""")
    st.markdown("""
Classifier-free guidance pushes the output further towards the hint. Because our 8×8 hint is so
small and noisy, guidance amplified the noise rather than the shape, so we kept guidance at 1.
You can try this yourself with the *Guidance strength* slider.

**Effect of noise** on diffusion shape accuracy: 0.83 (no noise), 0.77 (noise 0.1),
0.70 (noise 0.2).
""")

    st.header("Limitations")
    st.markdown("""
- The model only knows **stars and moons**. Anything else is redrawn as whichever it resembles.
- It makes a **plausible** doodle, not the original: the same input can give different results.
- Quick, Draw! sketches take under 20 seconds, so some training drawings are scribbles that
  aren't clearly either shape.
""")
    st.caption("Data: Google Creative Lab, *The Quick, Draw! Dataset* (CC BY 4.0). "
               "LS01 42.500 Data, Technology & Design.")


# ----------------------------------------------------------------------------- page
st.title("⭐🌙 Quick, Draw! Diffusion")
st.markdown(
    "Draw a **star** or a **moon**. The app shrinks your doodle to a tiny, noisy "
    "**8×8** image, then a **diffusion model** tries to redraw it as a sharp **28×28** "
    "doodle, starting from pure random noise and using the 8×8 image as its only clue.")

if not os.path.exists(MODEL_PATH):
    st.error("model.pt not found. Train it first with: `python train.py`")
    st.stop()

with st.sidebar:
    st.header("Settings")
    noise = st.slider("Noise added to the 8×8 input", 0.0, 0.3, 0.1, 0.05,
                      help="The brief's formula z = √a·x + √(1−a)·ε with a = 1 − noise. "
                           "The model was trained with noise up to 0.3.")
    steps = st.slider("Diffusion steps", 10, 200, 50, 10,
                      help="More steps is slower but can give cleaner strokes. "
                           "Our results use 50.")
    guidance = st.slider("Guidance strength", 1.0, 3.0, 1.0, 0.5,
                         help="Classifier-free guidance. 1 = off (what we use). Higher values "
                              "push harder towards the hint, but in our tests this amplified "
                              "the noise and made results worse; try it and see.")
    n_samples = st.slider("Number of extra samples", 2, 8, 4)
    seed = int(st.number_input("Random seed", 0, 9999, 42,
                               help="Same seed + same input = same result."))
    st.divider()
    st.caption("Model trained on 18,000 Quick, Draw! stars and moons. "
               "Other shapes will be squeezed into a star or a moon.")

tab_try, tab_about = st.tabs(["🎨 Try it", "📊 How it works & results"])

# The About tab is drawn first, because st.stop() in the Try tab ends the script.
with tab_about:
    render_about()

with tab_try:
    left, right = st.columns([1, 2], gap="large")

    with left:
        source = st.radio("Input", ["Draw", "Upload", "Example"], horizontal=True,
                          label_visibility="collapsed")
        image = None
        if source == "Draw":
            st.caption("Draw with your mouse, finger or stylus. Results update after each stroke.")
            data_url = draw_canvas(size=CANVAS_SIZE, pen=16, key="canvas", default=None)
            if data_url:
                image = decode_data_url(data_url)
        elif source == "Upload":
            upload = st.file_uploader("Upload a doodle (dark lines on a light background, "
                                      "or the other way round)",
                                      type=["png", "jpg", "jpeg", "webp"])
            if upload:
                image = Image.open(upload)
                st.image(image, caption="Your upload", width=CANVAS_SIZE)
        else:
            name = st.selectbox("Example", list(EXAMPLES))
            image = EXAMPLES[name]()
            st.image(image, width=CANVAS_SIZE)

    with right:
        if image is None:
            st.info("👈 Draw a star or a moon, upload a picture, or choose an example.")
            st.stop()

        x, blank = preprocess_image(image)
        if blank:
            st.warning("The image looks blank. Try a drawing with clear lines.")
            st.stop()

        with st.spinner("Removing noise, step by step..."):
            r = run_pipeline(x.numpy().tobytes(), noise, steps, guidance, seed, n_samples)
        cats = r["categories"]

        st.subheader("1 · From your doodle to the model's output")
        c = st.columns(4)
        show(c[0], r["x"], "Your doodle (28×28)")
        show(c[1], r["shrunk"], "Shrunk to 8×8")
        show(c[2], r["hint"], f"+ noise {noise:g}: the model's only input")
        show(c[3], r["outputs"]["Diffusion"], "Diffusion output (28×28)")
        if r["probs_outputs"] is not None:
            c[0].markdown(verdict(r["probs_input"], cats))
            c[3].markdown(verdict(r["probs_outputs"], cats, len(r["outputs"]) - 1))

        st.subheader("2 · Watch the noise turn into a doodle")
        g1, g2 = st.columns([1, 3])
        g1.image(frames_gif(r["frames"]), caption="Denoising (animated)")
        g2.image(frames_strip(r["frames"]), width="stretch",
                 caption="Pure noise  →  →  →  final doodle")
        g2.caption("The model starts from random noise and removes a little at each step, "
                   "guided by the noisy 8×8 input.")

        st.subheader("3 · Compare with the baselines")
        c = st.columns(len(r["outputs"]) + 1)
        show(c[0], r["x"], "Your doodle (target)")
        for i, (name, img) in enumerate(r["outputs"].items()):
            col = c[i + 1]
            show(col, img, name)
            col.markdown(f"MSE **{r['mse'][i]:.3f}** · SSIM **{r['ssim'][i]:.2f}**")
            if r["probs_outputs"] is not None:
                col.markdown(verdict(r["probs_outputs"], cats, i))
        st.caption("MSE and SSIM compare outputs with your doodle pixel by pixel, so blurry "
                   "outputs often score better. The shape classifier checks whether each "
                   "output actually looks like a star or a moon. Diffusion usually wins there.")

        st.subheader("4 · Same input, different results")
        c = st.columns(n_samples)
        for i, col in enumerate(c):
            show(col, r["samples"][i:i + 1], f"Sample {i + 1}", size=140)
            if r["probs_samples"] is not None:
                col.caption(verdict(r["probs_samples"], cats, i).replace("**", ""))
        st.caption("Diffusion generates a *plausible* doodle that fits the clue rather than "
                   "copying the original, so each run of random noise gives a different "
                   "answer. An 8×8 image simply doesn't contain enough detail to know the "
                   "exact original.")
