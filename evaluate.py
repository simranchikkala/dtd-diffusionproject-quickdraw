"""
This file is used to evaluate the performance of the selected diffusion model vs the two baselines CNN and LR. 
It takes 300 test images which were held back from training and makes noisy 8x8 images at 3 different noise levels. 
It runs 4 methods: Bicubic upsampling, Linear Regression, CNN and Diffusion U-Net to denoise the images and 
compares the results using MSE and SSIM metrics.

Usage: 
    python evaluate.py --n (300) --steps () 

Outputs:
    results/metrics.csv          MSE and SSIM per method, noise level and category
    results/comparison.png       example images side by side
    results/denoising_steps.png  diffusion going from static to a drawing
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")                     # no window needed; works on servers and CI
import matplotlib.pyplot as plt
import torch

from src.data import load_quickdraw, make_hint, train_test_split, upsample
from src.diffusion import Diffusion
from src.metrics import mse, ssim
from src.models import CNNSR, LinearSR, ShapeClassifier, UNet, load_checkpoint


def load_all(models_dir, model_path):
    linear, _ = load_checkpoint(os.path.join(models_dir, "linear.pt"), LinearSR)
    cnn, _ = load_checkpoint(os.path.join(models_dir, "cnn.pt"), CNNSR)
    clf, _ = load_checkpoint(os.path.join(models_dir, "classifier.pt"), ShapeClassifier)
    unet, ckpt = load_checkpoint(model_path, UNet)
    return linear, cnn, clf, unet, Diffusion(ckpt.get("timesteps", 200))


@torch.no_grad()
def shape_correct(clf, images, labels):
    """1.0 where the classifier recognises the image as its true category."""
    return (clf(images).argmax(dim=1) == labels).float()


@torch.no_grad()
def run_diffusion(diffusion, unet, hint, steps, guidance, seed, batch=100):
    out = [diffusion.sample(unet, hint[i:i + batch], steps=steps, guidance=guidance,
                            seed=seed + i) for i in range(0, len(hint), batch)]
    return torch.cat(out)


@torch.no_grad()
def predict_all(hint, linear, cnn, unet, diffusion, steps, guidance, seed):
    return {
        "bicubic": upsample(hint.clamp(0, 1)),
        "linear": linear(hint).clamp(0, 1),
        "cnn": cnn(hint),
        "diffusion": run_diffusion(diffusion, unet, hint, steps, guidance, seed),
    }


def show_grid(rows, col_titles, path, title, row_labels=None):
    """rows: list of lists of (1,28,28)/(1,8,8) tensors."""
    n_r, n_c = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_r, n_c, figsize=(n_c * 1.6, n_r * 1.6))
    axes = axes.reshape(n_r, n_c)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r, c]
            ax.imshow(img.squeeze().clamp(0, 1), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 and col_titles:
                ax.set_title(col_titles[c], fontsize=9)
            if c == 0 and row_labels:
                ax.set_ylabel(row_labels[r], fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/quickdraw.npz")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--model-path", default="model.pt")
    p.add_argument("--n", type=int, default=300, help="test images to evaluate")
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    p.add_argument("--steps", type=int, default=50, help="diffusion sampling steps")
    p.add_argument("--guidance", type=float, default=1.0,
                   help="classifier-free guidance strength (1 = none)")
    p.add_argument("--guidance-sweep", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0],
                   help="guidance strengths to compare; pass a single value to make it quicker")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images, labels, categories = load_quickdraw(args.data)
    _, test_idx = train_test_split(len(images), seed=args.seed)
    test_idx = test_idx[: args.n]
    x, y = images[test_idx], labels[test_idx]
    linear, cnn, clf, unet, diffusion = load_all(args.models_dir, args.model_path)
    methods = ["bicubic", "linear", "cnn", "diffusion"]
    groups = [("all", torch.ones_like(y, dtype=torch.bool))] + \
             [(c, y == k) for k, c in enumerate(categories)]

    # The classifier's accuracy on the real drawings is the best any method can get.
    real_acc = shape_correct(clf, x, y)
    print(f"Classifier accuracy on the real test drawings: {real_acc.mean():.3f} "
          f"(the ceiling for shape_acc)")

    rows, examples = [], {}
    for k, c in groups:
        rows.append(["real drawings", "-", k, 0.0, 1.0, real_acc[c].mean().item()])

    for level in args.noise_levels:
        g = torch.Generator().manual_seed(args.seed)
        hint = make_hint(x, level, g)
        preds = predict_all(hint, linear, cnn, unet, diffusion, args.steps,
                            args.guidance, args.seed)
        examples[level] = (hint, preds)

        print(f"\nNoise level {level}  (diffusion guidance {args.guidance})")
        print(f"  {'method':<10} {'category':<9} {'MSE':>7} {'SSIM':>7} {'shape':>7}")
        for m in methods:
            m_mse, m_ssim = mse(preds[m], x), ssim(preds[m], x)
            m_acc = shape_correct(clf, preds[m], y)
            for name, mask in groups:
                if mask.sum() == 0:
                    continue
                r = [m, level, name, m_mse[mask].mean().item(),
                     m_ssim[mask].mean().item(), m_acc[mask].mean().item()]
                rows.append(r)
                print(f"  {m:<10} {name:<9} {r[3]:7.4f} {r[4]:7.3f} {r[5]:7.3f}")

    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "noise_level", "category", "mse", "ssim", "shape_acc"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.5f}", f"{r[4]:.4f}", f"{r[5]:.4f}"])

    level = args.noise_levels[len(args.noise_levels) // 2]
    hint, preds = examples[level]

    # ---- Guidance sweep: does following the hint more strongly help?
    if args.guidance_sweep:
        print(f"\nGuidance sweep at noise level {level}")
        print(f"  {'guidance':>8} {'MSE':>7} {'SSIM':>7} {'shape':>7}")
        sweep_rows, sweep_imgs = [], {}
        for w_g in args.guidance_sweep:
            out = run_diffusion(diffusion, unet, hint, args.steps, w_g, args.seed)
            sweep_imgs[w_g] = out
            r = [w_g, mse(out, x).mean().item(), ssim(out, x).mean().item(),
                 shape_correct(clf, out, y).mean().item()]
            sweep_rows.append(r)
            print(f"  {r[0]:8.1f} {r[1]:7.4f} {r[2]:7.3f} {r[3]:7.3f}")
        with open(os.path.join(args.out_dir, "guidance.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["guidance", "noise_level", "mse", "ssim", "shape_acc"])
            for r in sweep_rows:
                w.writerow([r[0], level, f"{r[1]:.5f}", f"{r[2]:.4f}", f"{r[3]:.4f}"])
        n_show = min(6, len(x))
        show_grid([[hint[i]] + [sweep_imgs[w_g][i] for w_g in args.guidance_sweep] + [x[i]]
                   for i in range(n_show)],
                  ["8x8 input"] + [f"guidance {w_g:g}" for w_g in args.guidance_sweep]
                  + ["target"],
                  os.path.join(args.out_dir, "guidance.png"),
                  f"Effect of guidance strength (noise level {level})")

    # ---- Figure: side-by-side comparison
    n_show = min(6, len(x))
    show_grid([[hint[i]] + [preds[m][i] for m in methods] + [x[i]] for i in range(n_show)],
              ["8x8 input"] + methods + ["target"],
              os.path.join(args.out_dir, "comparison.png"),
              f"Super-resolution 8x8 -> 28x28 (noise level {level}, guidance {args.guidance:g})")

    # ---- Figure: several diffusion samples for the same input
    n_inputs, n_samples = min(5, len(x)), 4
    sample_rows = []
    for i in range(n_inputs):
        outs = [diffusion.sample(unet, hint[i:i + 1], steps=args.steps,
                                 guidance=args.guidance, seed=args.seed + 1000 * s)[0]
                for s in range(n_samples)]
        sample_rows.append([hint[i]] + outs + [x[i]])
    show_grid(sample_rows,
              ["8x8 input"] + [f"sample {s + 1}" for s in range(n_samples)] + ["target"],
              os.path.join(args.out_dir, "samples.png"),
              "Same input, different random starts: diffusion gives a plausible "
              "doodle, not the exact one")

    # ---- Figure: denoising steps
    _, frames = diffusion.sample(unet, hint[:4], steps=args.steps, guidance=args.guidance,
                                 seed=args.seed, n_frames=8)
    show_grid([[fr[r] for fr in frames] for r in range(4)], None,
              os.path.join(args.out_dir, "denoising_steps.png"),
              "Diffusion sampling: from pure noise (left) to drawing (right)")

    print(f"\nSaved results to {args.out_dir}/")


if __name__ == "__main__":
    main()

