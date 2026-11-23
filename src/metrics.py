"""
Evaluation metrics.

MSE  (Rewards blurrier guesses if they are considered to be "safe" )
SSIM - structural similarity. Evaluates based on edges and local constrast. 
"""

import torch
import torch.nn.functional as F


def mse(pred, target):
    """Per-image MSE. Inputs (B,1,H,W) in [0,1]. Returns tensor (B,)."""
    return ((pred - target) ** 2).flatten(1).mean(dim=1)


def _gaussian_window(size=7, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim(pred, target, data_range=1.0, window_size=7):
    """Per-image SSIM. Inputs (B,1,H,W) in [0,1]. Returns tensor (B,)."""
    w = _gaussian_window(window_size).to(pred.device, pred.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    mu_x = F.conv2d(pred, w)
    mu_y = F.conv2d(target, w)
    sxx = F.conv2d(pred * pred, w) - mu_x ** 2
    syy = F.conv2d(target * target, w) - mu_y ** 2
    sxy = F.conv2d(pred * target, w) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return ssim_map.flatten(1).mean(dim=1)