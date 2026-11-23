"""
This file contains code to train our diffusion model and two baselines using LR and CNNs. 
The baselines are used to compare the performance of our diffusion model.
The Diffusion model works by: 
       1. Loading Data ( Read data/quickdraw.npz ): Turn each image of a doodle into 
       0(black background) and 1(white stroke), each 28x28 pixels.
       2. Split the data into training and testing. 
       3. Make input hints. For each image, shrink the 28x28 image to 8x8 by averaging blocks of pixels.
          Brighten the 8x8 image for standardisation and add noise using the given formula ()
       4. Each model is trained using this cleaned and processed data. 
            A.Linear Regression Model: using LS method each output is a weighted sum of the input pixels.
            B.CNN Model: stretched 8x8 image to 28x28 and refines it using 4 layers of 3x3 filters. Using MSE as metric and 10 epochs as hyperparameter./
            C.Diffusion U-Net Model: At each step it picks random noise between 0-199 and adds the selected amount to the real 28x28 doodle.
                                     Then the U-Net model takes the noisy 28x28 image and the stretched hint and guesses the noise added and adjusts the image accordingly.
                                     The model is trained to minimize the MSE between the predicted noise and the actual noise added over 20 training epochs.


Usage: 
    python train.py 

Outputs:
    models/linear.pt    linear regression baseline
    models/cnn.pt       CNN baseline
    models/diffusion.pt conditional diffusion U-Net (used by the app)
    models/train_log.csv  loss per epoch (keep it for the report)
"""


import argparse
import copy
import csv
import os
import time

import torch
import torch.nn.functional as F

from src.data import load_quickdraw, make_hint, train_test_split
from src.diffusion import Diffusion
from src.models import CNNSR, LinearSR, ShapeClassifier, UNet, save_model


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def random_levels(n, max_noise, generator):
    """Different noise level per image, so the model copes with any slider value."""
    return torch.rand(n, generator=generator) * max_noise


def batches(n, batch_size, generator):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        yield perm[i: i + batch_size]


class Logger:
    def __init__(self, path, fresh=False):
        self.path = path
        new = fresh or not os.path.exists(path)
        self.f = open(path, "w" if fresh else "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["time", "model", "epoch", "train_loss", "seconds"])

    def log(self, model, epoch, loss, seconds):
        self.w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), model, epoch,
                         f"{loss:.6f}", f"{seconds:.1f}"])
        self.f.flush()
        print(f"  [{model}] epoch {epoch}: loss {loss:.5f} ({seconds:.0f}s)")


# --------------------------------------------------------------------------
def train_linear(x_train, args, g):
    """Closed-form ridge regression: weights = (X^T X + lambda I)^-1 X^T Y."""
    print("Training linear regression baseline...")
    x_rep = x_train.repeat(2, 1, 1, 1)               # 2 noisy copies of each image
    hints = make_hint(x_rep, random_levels(len(x_rep), args.max_noise, g), g)
    X = torch.cat([hints.flatten(1), torch.ones(len(x_rep), 1)], dim=1).double()
    Y = x_rep.flatten(1).double()
    A = X.T @ X + 1e-3 * torch.eye(X.shape[1], dtype=torch.float64)
    W = torch.linalg.solve(A, X.T @ Y)               # (65, 784)

    model = LinearSR()
    with torch.no_grad():
        model.linear.weight.copy_(W[:-1].T.float())
        model.linear.bias.copy_(W[-1].float())
    save_model(model, os.path.join(args.out_dir, "linear.pt"))
    print("  saved models/linear.pt")


def train_cnn(x_train, args, g, device, logger):
    print("Training CNN baseline...")
    model = CNNSR().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.cnn_epochs + 1):
        start, total, count = time.time(), 0.0, 0
        for idx in batches(len(x_train), args.batch_size, g):
            x = x_train[idx]
            hint = make_hint(x, random_levels(len(x), args.max_noise, g), g)
            loss = F.mse_loss(model(hint.to(device)), x.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, count = total + loss.item() * len(x), count + len(x)
        logger.log("cnn", epoch, total / count, time.time() - start)
    save_model(model.cpu(), os.path.join(args.out_dir, "cnn.pt"))
    print("  saved models/cnn.pt")


def train_diffusion(x_train, args, g, device, logger, categories):
    print("Training conditional diffusion U-Net...")
    diffusion = Diffusion(timesteps=args.timesteps)
    model = UNet(base=args.base).to(device)
    ema = copy.deepcopy(model).eval()        # smoothed weights give cleaner samples
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = (len(x_train) + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps_per_epoch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.2f}M parameters, device={device}")
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total, count = time.time(), 0.0, 0
        for idx in batches(len(x_train), args.batch_size, g):
            x = x_train[idx]
            hint = make_hint(x, random_levels(len(x), args.max_noise, g), g)
            loss = diffusion.loss(model, x.to(device), hint.to(device), args.p_uncond)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            decay = min(args.ema_decay, (1 + step) / (10 + step))  # warm-up for short runs
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
            total, count = total + loss.item() * len(x), count + len(x)
        logger.log("diffusion", epoch, total / count, time.time() - start)

        # Save every epoch so a long run can be stopped early and still be used.
        save_model(ema.cpu(), args.model_path, timesteps=args.timesteps,
                   max_noise=args.max_noise, p_uncond=args.p_uncond,
                   categories=categories, epochs_done=epoch)
        ema.to(device)
    print(f"  saved {args.model_path}")


def train_classifier(x_train, y_train, args, g, device, logger, categories):
    """A judge for evaluate.py: does a generated image look like the right category?"""
    print("Training shape classifier...")
    model = ShapeClassifier(n_classes=len(categories)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.clf_epochs + 1):
        model.train()
        start, total, correct, count = time.time(), 0.0, 0, 0
        for idx in batches(len(x_train), args.batch_size, g):
            x, y = x_train[idx], y_train[idx]
            # Small random shifts so it judges shape, not exact position.
            dx, dy = torch.randint(-2, 3, (2,), generator=g).tolist()
            x = torch.roll(x, shifts=(dy, dx), dims=(2, 3)).to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(x)
        logger.log("classifier", epoch, total / count, time.time() - start)
        print(f"    train accuracy {correct / count:.3f}")
    save_model(model.cpu(), os.path.join(args.out_dir, "classifier.pt"),
               categories=categories)
    print("  saved models/classifier.pt")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/quickdraw.npz")
    p.add_argument("--only", choices=["all", "linear", "cnn", "classifier", "diffusion"], default="all")
    p.add_argument("--epochs", type=int, default=20, help="diffusion epochs")
    p.add_argument("--cnn-epochs", type=int, default=10)
    p.add_argument("--clf-epochs", type=int, default=5, help="shape classifier epochs")
    p.add_argument("--p-uncond", type=float, default=0.1,
                   help="fraction of training images whose hint is hidden (for guidance)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32, help="U-Net width")
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--max-noise", type=float, default=0.3,
                   help="largest noise level added to the 8x8 input during training")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--limit", type=int, default=0, help="use only N training images (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--model-path", default="model.pt")
    args = p.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"{args.data} not found. Run: python download_data.py")

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    images, labels, categories = load_quickdraw(args.data)
    train_idx, _ = train_test_split(len(images), seed=args.seed)
    x_train, y_train = images[train_idx], labels[train_idx]
    if args.limit:
        x_train, y_train = x_train[: args.limit], y_train[: args.limit]
    print(f"Categories: {categories} | training images: {len(x_train)}")

    logger = Logger(os.path.join(args.out_dir, "train_log.csv"), fresh=args.only == "all")
    if args.only in ("all", "linear"):
        train_linear(x_train, args, g)
    if args.only in ("all", "cnn"):
        train_cnn(x_train, args, g, device, logger)
    if args.only in ("all", "classifier"):
        train_classifier(x_train, y_train, args, g, device, logger, categories)
    if args.only in ("all", "diffusion"):
        train_diffusion(x_train, args, g, device, logger, categories)
    print("Done.")


if __name__ == "__main__":
    main()
