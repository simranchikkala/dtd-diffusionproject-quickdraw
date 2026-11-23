"""
The four methods we compare, from simplest to best:

    1. bicubic upscaling        (no learning; in src/data.py)
    2. LinearSR                 linear regression 8x8 -> 28x28 (brief's baseline)
    3. CNNSR                    small CNN, one pass 8x8 -> 28x28
    4. UNet + diffusion         conditional diffusion super-resolution (main model)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import HIGH_RES, LOW_RES

class LinearSR(nn.Module):
    """Linear regression: every output pixel is a weighted sum of the 64 input pixels.

    It has no notion that neighbouring pixels are related, which is the
    limitation that motivates the CNN.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(LOW_RES * LOW_RES, HIGH_RES * HIGH_RES)

    def forward(self, hint):                       # hint: (B, 1, 8, 8)
        out = self.linear(hint.flatten(1))
        return out.view(-1, 1, HIGH_RES, HIGH_RES)


class CNNSR(nn.Module):
    """Small CNN: upscale, then refine with 3x3 convolutions (local spatial structure)."""

    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, hint):                       # hint: (B, 1, 8, 8)
        up = F.interpolate(hint, size=(HIGH_RES, HIGH_RES), mode="bilinear",
                           align_corners=False)
        return torch.sigmoid(self.net(up))


def timestep_embedding(t, dim):
    """Sinusoidal embedding so the network knows which noise level it is at."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Predicts the noise in a noisy 28x28 image, given the upscaled 8x8 hint.

    Inputs:  x_t      (B, 1, 28, 28)  noisy high-res image, values roughly in [-1, 1]
             cond     (B, 1, 28, 28)  upscaled hint, in [-1, 1]
             t        (B,)            integer timestep
             has_hint (B,)            1 = hint given, 0 = hint hidden. Hiding the hint
                                      sometimes during training enables classifier-free
                                      guidance (see src/diffusion.py).
    Output:  predicted noise, (B, 1, 28, 28)

    Resolutions: 28 -> 14 -> 7 -> 14 -> 28
    """

    def __init__(self, base=32):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        temb_dim = base * 4
        self.config = {"base": base}

        self.temb = nn.Sequential(
            nn.Linear(base, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.base = base

        self.inp = nn.Conv2d(3, c1, 3, padding=1)  # 3 channels: x_t, cond, hint flag
        self.down1 = ResBlock(c1, c1, temb_dim)            # 28
        self.pool1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)   # -> 14
        self.down2 = ResBlock(c1, c2, temb_dim)            # 14
        self.pool2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)   # -> 7
        self.mid1 = ResBlock(c2, c3, temb_dim)             # 7
        self.mid2 = ResBlock(c3, c3, temb_dim)
        self.up2 = ResBlock(c3 + c2, c2, temb_dim)         # 14
        self.up1 = ResBlock(c2 + c1, c1, temb_dim)         # 28
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(),
                                 nn.Conv2d(c1, 1, 3, padding=1))

    def forward(self, x_t, cond, t, has_hint=None):
        if has_hint is None:
            has_hint = torch.ones(x_t.shape[0], device=x_t.device)
        flag = has_hint.float().view(-1, 1, 1, 1)
        temb = self.temb(timestep_embedding(t, self.base))
        h = self.inp(torch.cat([x_t, cond * flag, flag.expand_as(x_t)], dim=1))
        s1 = self.down1(h, temb)                            # 28
        s2 = self.down2(self.pool1(s1), temb)               # 14
        h = self.mid2(self.mid1(self.pool2(s2), temb), temb)  # 7
        h = F.interpolate(h, scale_factor=2, mode="nearest")  # 14
        h = self.up2(torch.cat([h, s2], dim=1), temb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")  # 28
        h = self.up1(torch.cat([h, s1], dim=1), temb)
        return self.out(h)



class ShapeClassifier(nn.Module):
    """Small CNN that says which category a 28x28 doodle is (e.g. star or moon).

    Used as a 'judge': if a generated image is classified as the same category as
    the original drawing, the model drew the right kind of shape. This measures
    something MSE and SSIM can't: whether the output is the right thing at all.
    """

    def __init__(self, n_classes=2):
        super().__init__()
        self.config = {"n_classes": n_classes}
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 7
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(64, n_classes),
        )

    def forward(self, x):                          
        return self.net(x)



def save_model(model, path, **extra):
    """Save weights plus a small config dict (safe to load with weights_only=True)."""
    torch.save({"state_dict": model.state_dict(),
                "config": getattr(model, "config", {}),
                **extra}, path)


def load_checkpoint(path, model_cls, map_location="cpu"):
    """Load a model saved by save_model. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = model_cls(**ckpt.get("config", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
