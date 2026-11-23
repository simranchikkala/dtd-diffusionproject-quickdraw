"""
Conditional diffusion for super-resolution.

Forward (noising) process, the brief's formula:
    z_t = sqrt(a_t) * x + sqrt(1 - a_t) * eps
where a_t (often written alpha-bar_t) shrinks from ~1 (almost clean) at t=0
to ~0 (almost pure noise) at t=T-1.

Training (Ho et al., 2020): pick a random t, noise the 28x28 target, and teach
the U-Net to predict eps, given z_t and the upscaled 8x8 hint.

Sampling: start from pure noise and remove it step by step. We use the DDIM
update (Song et al., 2021), which lets us take fewer steps (e.g. 50 instead of
200) so the app responds quickly. With eta=0 it is deterministic for a fixed seed.

Classifier-free guidance (Ho & Salimans, 2022): during training the hint is
hidden for a random 10% of images, so the same network also learns to draw
*any* doodle without a hint. When sampling we run both versions and push the
result away from 'any doodle' towards 'the doodle this hint describes':

    eps = eps_without_hint + w * (eps_with_hint - eps_without_hint)

w = 1 is plain conditioning; w > 1 makes the output follow the hint more closely.
"""

import torch
import torch.nn.functional as F

from src.data import HIGH_RES, upsample


def to_model_range(x):      # [0, 1] -> [-1, 1]
    return x * 2.0 - 1.0


def from_model_range(x):    # [-1, 1] -> [0, 1]
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


class Diffusion:
    def __init__(self, timesteps=200):
        self.T = timesteps
        # Linear schedule, rescaled so T=200 adds as much noise as the
        # standard 1000-step schedule (beta 1e-4 -> 0.02).
        scale = 1000 / timesteps
        betas = torch.linspace(1e-4 * scale, 0.02 * scale, timesteps)
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0)   # the brief's a_t

    # ---------------------------------------------------------------- training
    def q_sample(self, x0, t, noise):
        """Noise x0 to timestep t using the brief's formula."""
        a = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    def loss(self, model, x_high, hint, p_uncond=0.1):
        """One training loss. x_high: (B,1,28,28) in [0,1]; hint: (B,1,8,8).

        p_uncond: fraction of images whose hint is hidden (for guidance).
        """
        x0 = to_model_range(x_high)
        cond = to_model_range(upsample(hint.clamp(0, 1)))
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        has_hint = (torch.rand(b, device=x0.device) >= p_uncond).float()
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), cond, t, has_hint)
        return F.mse_loss(pred, noise)

    def predict_noise(self, model, x, cond, t, guidance):
        """Noise prediction with classifier-free guidance (one batched forward pass)."""
        b = x.shape[0]
        tt = torch.full((b,), t, device=x.device, dtype=torch.long)
        if guidance == 1.0:
            return model(x, cond, tt)
        ones, zeros = torch.ones(b, device=x.device), torch.zeros(b, device=x.device)
        eps = model(torch.cat([x, x]), torch.cat([cond, cond]), torch.cat([tt, tt]),
                    torch.cat([ones, zeros]))
        eps_cond, eps_uncond = eps.chunk(2)
        return eps_uncond + guidance * (eps_cond - eps_uncond)

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, model, hint, steps=50, guidance=1.0, eta=0.0, seed=None, n_frames=0):
        """Generate 28x28 images from 8x8 hints.

        hint:     (B, 1, 8, 8) noisy low-res input, roughly in [0, 1]
        steps:    number of denoising steps (<= T); fewer is faster
        guidance: how strongly to follow the hint. 1 = no extra push (best for our
                  model in testing); higher values over-sharpen into scribbles
        eta:      0 = deterministic DDIM, 1 = DDPM-like random steps
        seed:     fix this for repeatable results
        n_frames: also return this many in-between images for an animation

        Returns images in [0, 1] (B, 1, 28, 28), and a list of frames if n_frames > 0.
        """
        device = next(model.parameters()).device
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)

        cond = to_model_range(upsample(hint.clamp(0, 1))).to(device)
        b = hint.shape[0]
        x = torch.randn((b, 1, HIGH_RES, HIGH_RES), generator=g).to(device)

        steps = max(1, min(steps, self.T))
        ts = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        alpha_bar = self.alpha_bar.to(device)
        keep = set()
        if n_frames > 0:
            keep = set(torch.linspace(0, steps - 1, n_frames).round().long().tolist())
        frames = [from_model_range(x).cpu()] if n_frames > 0 else []

        for i, t in enumerate(ts):
            a_t = alpha_bar[t]
            a_prev = alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            eps = self.predict_noise(model, x, cond, t, guidance)

            x0_pred = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp(min=0).sqrt()
            direction = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            x = a_prev.sqrt() * x0_pred + direction
            if eta > 0 and i + 1 < len(ts):
                x = x + sigma * torch.randn(x.shape, generator=g).to(device)

            if i in keep and i != len(ts) - 1:
                frames.append(from_model_range(x).cpu())

        result = from_model_range(x).cpu()
        if n_frames > 0:
            frames.append(result)
            return result, frames
        return result
