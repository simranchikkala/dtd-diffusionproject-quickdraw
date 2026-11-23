# Quick, Draw! Diffusion: Doodle Super-Resolution

This project has been completed and compiled by the following students:

| Name | Student ID | Main responsibilities |
|---|---|---|
| Aditi Raj | _ID_ | |
| Joe| _ID_ |  |
| Rakshita | _ID_ |  |
| Simran Srinivasan Chikkala | 1012380 |  |

as a part of the completion of the course LS01 42.500 Data, Technology & Design.

This project demonstrates understanding and application of the following topics: 
        1. Diffusion Models and their working. 
        2. Application of Diffusion Models as image generating and enhancing solutions. 
        3. Comparison of various techniques to design a solution to a given problem. 
        4. Standardisation of running a model across multiple devices using tools such as GitHub and Docker.
        5. Automated testing and continuous integration, so every code change is checked automatically.
        6. Deployment of a machine learning model as a web application that other people can use.

---

## Problem and approach

The course brief reconstructs a noisy 8×8 image of the digit "6". We extend this idea to **super-resolution**: given a small, noisy **8×8** version of a doodle, the model generates a sharp **28×28** version of it.

How the pipeline works:

1. A 28×28 doodle is shrunk to 8×8, which loses most of its detail.
2. Noise is added to the 8×8 image using the brief's formula, `z = sqrt(a) * x + sqrt(1 - a) * eps`.
3. A **conditional diffusion model** starts from pure random noise at 28×28 and removes it step by step, using the noisy 8×8 image as a guide, until a clear doodle appears.

Following the brief's *Baseline → Evaluate → Identify Limitation → Improve Model* approach, we compare four methods:

| Method | Idea |
|---|---|
| Bicubic upscaling | Stretches the image; no learning |
| Linear regression | Each output pixel is a weighted sum of the input pixels (the brief's baseline) |
| CNN | Small convolutional network that uses local spatial structure |
| Conditional diffusion (U-Net) | Generates the image from noise, guided by the 8×8 input |

## Dataset

The dataset selected was **Google's Quick, Draw!** for the categories **star** and **moon**. Quick, Draw! contains millions of doodles drawn by people around the world in under 20 seconds each, released by Google under the CC BY 4.0 licence. We use Google's pre-made 28×28 grayscale bitmaps (white strokes on a black background) and keep a random subset of **10,000 drawings per category** (20,000 in total), split 90% for training and 10% for testing with a fixed random seed.

The raw data is not stored in this repository because the files are large (about 100 MB per category). Anyone can re-create the exact same subset by running `download_data.py`.

---

## Project structure

This project contains the following files, each with a specific function:

| File / folder | Function |
|---|---|
| `download_data.py` | Downloads the chosen Quick, Draw! categories and saves a fixed random subset to `data/quickdraw.npz`. Uses only built-in Python, so it works before other packages are installed. |
| `train.py` | Trains the linear regression, CNN and diffusion models and saves them. Logs the loss for each epoch to `models/train_log.csv`. |
| `evaluate.py` | Compares all four methods on the test set using MSE and SSIM, and saves the results table and figures to `results/`. |
| `model.pt` | The trained diffusion model used by the app. |
| `models/` | The trained baseline models (`linear.pt`, `cnn.pt`) and the training log. |
| `src/data.py` | Loading data, shrinking images, adding noise and preprocessing user images. |
| `src/models.py` | The linear regression, CNN and U-Net model definitions. |
| `src/diffusion.py` | The noise schedule, training loss and step-by-step image generation. |
| `src/metrics.py` | MSE and SSIM evaluation metrics. |
| `requirements.txt` | Python packages needed to run the project. |
| `app.py` | _To be added:_ Streamlit web app. |
| `tests/` | _To be added:_ automated tests (pytest). |
| `Dockerfile` | _To be added:_ packages the app and its environment. |
| `.github/workflows/test.yml` | _To be added:_ runs the tests automatically on every push. |

---

## Installation

The project needs **Python 3.11–3.13** (we used 3.12).

```bash
git clone <repository-url>
cd <repository-folder>
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

**Windows users:** we recommend running the project inside **WSL (Ubuntu)**. On some Windows 11 laptops, Smart App Control blocks PyTorch's unsigned files, which causes a `WinError 4551` error when importing `torch`. The same commands above work inside WSL. **Google Colab** is another option and provides a free GPU for faster training.

## Usage

**1. Download the data**
```bash
python3 download_data.py
```
Options: `--n 5000` keeps fewer drawings per category; `--categories star moon cup` chooses other categories.

**2. Train the models**
```bash
python3 train.py
```
This takes about 35–45 minutes on a laptop CPU, or a few minutes on a GPU. For a quick check that everything works:
```bash
python3 train.py --epochs 2 --cnn-epochs 1 --limit 2000
```

**3. Evaluate**
```bash
python3 evaluate.py
```
Outputs:
- `results/metrics.csv`: MSE and SSIM for each method, noise level and category
- `results/comparison.png`: example results side by side
- `results/denoising_steps.png`: the diffusion model turning noise into a doodle

**4. Run the app**
_To be added._

## Running the tests

_To be added._

## Docker

_To be added._

## Automated checks (GitHub Actions)

_To be added._

## Live app

_To be added: link to the deployed app._

---

## Results

_
Results on 300 held-out test drawings (150 stars, 150 moons), with noise level 0.1 added to the 8×8 input. Diffusion uses guidance 1 and 50 sampling steps.

| Method | MSE ↓ | SSIM ↑ | Shape accuracy ↑ |
|---|---|---|---|
| Bicubic | 0.117 | 0.208 | 0.51 |
| Linear regression | **0.064** | 0.212 | 0.51 |
| CNN | 0.066 | **0.221** | 0.52 |
| Diffusion | 0.128 | 0.170 | **0.77** |
| Real drawings (ceiling) | 0 | 1 | 0.98 |

**Shape accuracy by category:** the baselines recognise 0–1% of stars and 100% of moons, because their blurry outputs all look like moons. Diffusion recognises 82% of stars and 71% of moons.

**Effect of noise on diffusion shape accuracy:** 0.83 (no noise), 0.77 (noise 0.1), 0.70 (noise 0.2).

**Effect of guidance strength** (diffusion, noise 0.1):

| Guidance | MSE ↓ | SSIM ↑ | Shape accuracy ↑ |
|---|---|---|---|
| **1 (used)** | **0.128** | **0.170** | **0.77** |
| 1.5 | 0.157 | 0.135 | 0.63 |
| 2 | 0.207 | 0.087 | 0.49 |
| 3 | 0.281 | 0.033 | 0.50 |

### What the results show

- The linear regression and CNN baselines score best on the pixel-based metrics (MSE and SSIM), because blurry outputs "hedge" between possible stroke positions. But they never produce a recognisable star.
- The diffusion model produces sharp doodles of the right shape far more often (77% vs about 51%). Its MSE is higher because a sharp stroke drawn slightly out of place is penalised more than a blur.
- Stronger classifier-free guidance made every metric worse: the 8×8 hint is so small and noisy that guidance amplified noise rather than shape.
- Diffusion produces *a* plausible doodle rather than *the* exact original: the same input can give different results from different random starts (`results/samples.png`).
---

## References

1. Google Creative Lab, "The Quick, Draw! Dataset," https://github.com/googlecreativelab/quickdraw-dataset
