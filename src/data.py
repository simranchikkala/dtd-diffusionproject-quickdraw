"""
Data helpers: loading Quick, Draw!, making low-res/high-res pairs,
adding noise (the brief's formula) and preprocessing user drawings.

Image convention everywhere in this project:
    float32 tensors of shape (B, 1, H, W), values in [0, 1],
    white strokes (1.0) on a black background (0.0).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HIGH_RES = 28   # target size (Quick, Draw! bitmaps are 28x28)
LOW_RES = 8     # input size (as in the brief's 8x8 example)


def load_quickdraw(path="data/quickdraw.npz"):
    """Return (images, labels, categories).

    images: float tensor (N, 1, 28, 28) in [0, 1]
    labels: long tensor (N,)
    categories: list of str
    """
    data = np.load(path)
    images = torch.from_numpy(data["images"]).float().div(255.0).unsqueeze(1)
    labels = torch.from_numpy(data["labels"]).long()
    categories = [str(c) for c in data["categories"]]
    return images, labels, categories


def train_test_split(n, test_frac=0.1, seed=42):
    """Return (train_idx, test_idx) as index tensors. Same seed -> same split."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = int(round(n * test_frac))
    return perm[n_test:], perm[:n_test]



def downsample(x, size=LOW_RES):
    """28x28 -> 8x8 by averaging pixel areas."""
    return F.interpolate(x, size=(size, size), mode="area")


def upsample(x, size=HIGH_RES, mode="bicubic"):
    """8x8 -> 28x28. Bicubic is also our no-learning baseline."""
    out = F.interpolate(x, size=(size, size), mode=mode, align_corners=False)
    return out.clamp(0.0, 1.0)



def add_noise(x, noise_level, generator=None):
    """z = sqrt(alpha) * x + sqrt(1 - alpha) * eps, with alpha = 1 - noise_level.

    noise_level: float in [0, 1) or a tensor of shape (B,) for per-image levels.
                 0 means no noise.
    """
    if not torch.is_tensor(noise_level):
        noise_level = torch.full((x.shape[0],), float(noise_level))
    alpha = (1.0 - noise_level).to(x.dtype).view(-1, 1, 1, 1)
    eps = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    return alpha.sqrt() * x + (1.0 - alpha).sqrt() * eps


def normalise(x, eps=1e-6):
    """Stretch each image so its brightest pixel is 1.

    Shrinking thin pen strokes to 8x8 leaves them faint grey (~0.3), which even
    mild noise would drown out. Stretching keeps the shape visible.
    """
    peak = x.flatten(1).amax(dim=1).clamp(min=eps).view(-1, 1, 1, 1)
    return (x / peak).clamp(0.0, 1.0)


def make_hint(x_high, noise_level, generator=None):
    """High-res image -> noisy 8x8 observation (what the user 'gives' the model)."""
    return add_noise(normalise(downsample(x_high)), noise_level, generator)


def preprocess_image(img, threshold=0.1, content_size=24):
    """Turn any user image into a Quick, Draw!-style 28x28 tensor.

    Handles: colour or transparent images, white backgrounds (inverted),
    any size or aspect ratio, off-centre drawings.

    Returns (tensor of shape (1, 1, 28, 28), is_blank).
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))

    # Transparent parts become white, then convert to grayscale.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    # Quick, Draw! is white-on-black; most uploads are black-on-white.
    if arr.mean() > .5:
        arr = 1.0 - arr

    # Crop to the drawing itself.
    ys, xs = np.where(arr > threshold)
    if len(ys) == 0:
        return torch.zeros(1, 1, HIGH_RES, HIGH_RES), True
    arr = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    # Scale the longest side to `content_size` and centre it in 28x28.
    h, w = arr.shape
    scale = content_size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (HIGH_RES, HIGH_RES), 0)
    canvas.paste(small, ((HIGH_RES - new_w) // 2, (HIGH_RES - new_h) // 2))

    out = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)
    return out.view(1, 1, HIGH_RES, HIGH_RES), False


def to_display(x, size=196):
    """Tensor (1,1,H,W) or (H,W) in [0,1] -> enlarged uint8 PIL image for showing."""
    arr = x.detach().squeeze().clamp(0, 1).cpu().numpy()
    img = Image.fromarray((arr * 255).astype(np.uint8))
    return img.resize((size, size), Image.Resampling.NEAREST)
