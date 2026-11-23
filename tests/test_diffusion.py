"""
Automated tests. Run with:  python -m pytest

They answer one question: after changing the code, does the important
functionality still work? They use small fake images, so they run in seconds
and don't need the dataset. The last group checks the real trained model and
is skipped if model.pt hasn't been created yet.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import (HIGH_RES, LOW_RES, add_noise, downsample, make_hint, normalise,
                      preprocess_image, train_test_split, upsample)
from src.diffusion import Diffusion
from src.metrics import mse, ssim
from src.models import (CNNSR, LinearSR, ShapeClassifier, UNet, load_checkpoint,
                        save_model)

torch.manual_seed(0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model.pt")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fake_drawings(n=4):
    """White strokes on black, like Quick, Draw!: a plus sign in each image."""
    x = torch.zeros(n, 1, HIGH_RES, HIGH_RES)
    x[:, :, 12:16, 4:24] = 1.0
    x[:, :, 4:24, 12:16] = 1.0
    return x


def small_unet():
    return UNet(base=8).eval()        # tiny version so tests run fast


# --------------------------------------------------------------------------
# 1. Image sizes and noise
# --------------------------------------------------------------------------
def test_downsample_gives_8x8():
    assert downsample(fake_drawings()).shape == (4, 1, LOW_RES, LOW_RES)


def test_upsample_gives_28x28_in_range():
    out = upsample(torch.rand(4, 1, LOW_RES, LOW_RES))
    assert out.shape == (4, 1, HIGH_RES, HIGH_RES)
    assert out.min() >= 0 and out.max() <= 1


def test_adding_noise_keeps_8x8_shape():
    x = torch.rand(4, 1, LOW_RES, LOW_RES)
    assert add_noise(x, 0.2).shape == (4, 1, LOW_RES, LOW_RES)


def test_zero_noise_returns_the_input():
    x = torch.rand(4, 1, LOW_RES, LOW_RES)
    assert torch.allclose(add_noise(x, 0.0), x)


def test_noise_follows_the_brief_formula():
    """z = sqrt(alpha) * x + sqrt(1 - alpha) * eps, with alpha = 1 - noise_level."""
    x = torch.rand(2, 1, LOW_RES, LOW_RES)
    z = add_noise(x, 0.3, torch.Generator().manual_seed(1))
    eps = torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(z, 0.7 ** 0.5 * x + 0.3 ** 0.5 * eps, atol=1e-6)


def test_same_seed_gives_same_noise():
    x = torch.rand(2, 1, LOW_RES, LOW_RES)
    a = add_noise(x, 0.2, torch.Generator().manual_seed(5))
    b = add_noise(x, 0.2, torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def test_normalise_brightest_pixel_is_one():
    x = torch.rand(3, 1, LOW_RES, LOW_RES) * 0.3
    assert torch.allclose(normalise(x).flatten(1).amax(1), torch.ones(3))


def test_normalise_blank_image_does_not_crash():
    out = normalise(torch.zeros(1, 1, LOW_RES, LOW_RES))
    assert torch.isfinite(out).all() and out.max() == 0


def test_make_hint_gives_noisy_8x8():
    hint = make_hint(fake_drawings(), 0.1)
    assert hint.shape == (4, 1, LOW_RES, LOW_RES)
    assert torch.isfinite(hint).all()


def test_train_test_split_is_repeatable_and_separate():
    train1, test1 = train_test_split(100, seed=42)
    train2, test2 = train_test_split(100, seed=42)
    assert torch.equal(test1, test2)                  # same split every time
    assert len(set(train1.tolist()) & set(test1.tolist())) == 0   # no overlap
    assert len(train1) + len(test1) == 100


# --------------------------------------------------------------------------
# 2. Models produce the expected output shapes
# --------------------------------------------------------------------------
def test_linear_model_output_shape():
    assert LinearSR()(torch.rand(4, 1, 8, 8)).shape == (4, 1, 28, 28)


def test_cnn_output_shape_and_range():
    out = CNNSR()(torch.rand(4, 1, 8, 8))
    assert out.shape == (4, 1, 28, 28)
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize("has_hint", [None, torch.ones(4), torch.zeros(4)])
def test_unet_output_shape(has_hint):
    t = torch.randint(0, 200, (4,))
    out = small_unet()(torch.randn(4, 1, 28, 28), torch.randn(4, 1, 28, 28), t, has_hint)
    assert out.shape == (4, 1, 28, 28)


def test_classifier_output_shape():
    assert ShapeClassifier(n_classes=2)(torch.rand(4, 1, 28, 28)).shape == (4, 2)


def test_save_and_load_gives_same_model(tmp_path):
    model = small_unet()
    path = tmp_path / "m.pt"
    save_model(model, path, timesteps=200)
    loaded, ckpt = load_checkpoint(path, UNet)
    assert ckpt["timesteps"] == 200
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------
# 3. Diffusion process
# --------------------------------------------------------------------------
def test_noise_schedule_goes_from_clean_to_pure_noise():
    a = Diffusion(200).alpha_bar
    assert (a[1:] < a[:-1]).all()          # always more noise at later steps
    assert a[0] > 0.99 and a[-1] < 0.01    # starts almost clean, ends almost pure noise


def test_diffusion_noising_follows_the_brief_formula():
    """z_t = sqrt(a_t) * x + sqrt(1 - a_t) * eps, at a few different timesteps."""
    d = Diffusion(200)
    x, eps = torch.rand(3, 1, 28, 28), torch.randn(3, 1, 28, 28)
    t = torch.tensor([0, 100, 199])
    a = d.alpha_bar[t].view(-1, 1, 1, 1)
    assert torch.allclose(d.q_sample(x, t, eps), a.sqrt() * x + (1 - a).sqrt() * eps, atol=1e-6)


def test_training_loss_is_a_finite_number():
    loss = Diffusion(200).loss(small_unet(), fake_drawings(), make_hint(fake_drawings(), 0.1))
    assert loss.dim() == 0 and torch.isfinite(loss)


@pytest.mark.parametrize("guidance", [1.0, 2.0])
def test_generated_images_are_valid(guidance):
    out = Diffusion(200).sample(small_unet(), torch.rand(3, 1, 8, 8), steps=5,
                                guidance=guidance, seed=0)
    assert out.shape == (3, 1, 28, 28)
    assert torch.isfinite(out).all()       # no NaN or infinity
    assert out.min() >= 0 and out.max() <= 1


def test_same_seed_gives_same_generated_image():
    d, model, hint = Diffusion(200), small_unet(), torch.rand(2, 1, 8, 8)
    a = d.sample(model, hint, steps=5, seed=7)
    b = d.sample(model, hint, steps=5, seed=7)
    c = d.sample(model, hint, steps=5, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_sampling_returns_requested_animation_frames():
    _, frames = Diffusion(200).sample(small_unet(), torch.rand(1, 1, 8, 8), steps=10,
                                      seed=0, n_frames=4)
    assert 2 <= len(frames) <= 5
    assert all(f.shape == (1, 1, 28, 28) for f in frames)


# --------------------------------------------------------------------------
# 4. Metrics
# --------------------------------------------------------------------------
def test_mse_is_zero_for_identical_images_and_positive_otherwise():
    x = fake_drawings()
    assert torch.allclose(mse(x, x), torch.zeros(4))
    assert (mse(x, 1 - x) > 0).all()


def test_ssim_is_one_for_identical_images_and_lower_with_noise():
    x = fake_drawings()
    noisy = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)
    assert torch.allclose(ssim(x, x), torch.ones(4), atol=1e-4)
    assert (ssim(x, noisy) < 0.99).all()


# --------------------------------------------------------------------------
# 5. Preprocessing user uploads (what the app relies on)
# --------------------------------------------------------------------------
def black_on_white(size=(200, 200), box=(60, 60, 140, 140), mode="RGB"):
    """A typical user drawing: black square outline on a white page."""
    img = Image.new(mode, size, "white")
    from PIL import ImageDraw
    ImageDraw.Draw(img).rectangle(box, outline="black", width=12)
    return img


def check_valid(out):
    assert out.shape == (1, 1, 28, 28)
    assert out.min() >= 0 and out.max() <= 1
    assert torch.isfinite(out).all()


def test_colour_image_on_white_background_is_inverted():
    out, blank = preprocess_image(black_on_white())
    check_valid(out)
    assert not blank
    assert out.mean() < 0.5        # mostly black background, like Quick, Draw!
    assert out.max() > 0.8         # with bright strokes


def test_already_white_on_black_is_not_inverted():
    img = Image.fromarray((fake_drawings(1)[0, 0].numpy() * 255).astype(np.uint8))
    out, _ = preprocess_image(img)
    assert out.mean() < 0.5


def test_transparent_png_works():
    img = Image.new("RGBA", (150, 150), (0, 0, 0, 0))   # fully transparent
    from PIL import ImageDraw
    ImageDraw.Draw(img).ellipse((30, 30, 120, 120), outline=(0, 0, 0, 255), width=10)
    out, blank = preprocess_image(img)
    check_valid(out)
    assert not blank and out.mean() < 0.5


@pytest.mark.parametrize("size", [(100, 37), (37, 100), (1000, 800), (28, 28), (9, 9)])
def test_any_image_size_becomes_28x28(size):
    w, h = size
    out, _ = preprocess_image(black_on_white(size, (w // 4, h // 4, 3 * w // 4, 3 * h // 4)))
    check_valid(out)


def test_grayscale_and_numpy_inputs_work():
    out1, _ = preprocess_image(black_on_white(mode="L"))
    out2, _ = preprocess_image(np.asarray(black_on_white()))
    check_valid(out1)
    check_valid(out2)


def test_blank_image_is_flagged():
    out, blank = preprocess_image(Image.new("RGB", (200, 200), "white"))
    assert blank
    assert out.shape == (1, 1, 28, 28) and out.max() == 0


def test_off_centre_drawing_is_centred():
    out, _ = preprocess_image(black_on_white(box=(5, 5, 50, 50)))   # top-left corner
    ys, xs = torch.nonzero(out[0, 0] > 0.3, as_tuple=True)
    assert abs(ys.float().mean() - 13.5) < 3 and abs(xs.float().mean() - 13.5) < 3


# --------------------------------------------------------------------------
# 6. The real trained model (skipped until model.pt exists)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model.pt not trained yet")
def test_trained_model_loads_and_generates():
    model, ckpt = load_checkpoint(MODEL_PATH, UNet)
    assert "timesteps" in ckpt
    hint = make_hint(fake_drawings(2), 0.1, torch.Generator().manual_seed(0))
    out = Diffusion(ckpt["timesteps"]).sample(model, hint, steps=10, seed=0)
    assert out.shape == (2, 1, 28, 28)
    assert torch.isfinite(out).all() and out.min() >= 0 and out.max() <= 1