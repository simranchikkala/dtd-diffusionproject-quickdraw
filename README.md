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

_To be filled in from `results/metrics.csv` after the final training run._

| Method | MSE (noise 0.1) ↓ | SSIM (noise 0.1) ↑ |
|---|---|---|
| Bicubic | | |
| Linear regression | | |
| CNN | | |
| Diffusion | | |

MSE measures average pixel error and tends to reward safe, blurry outputs. SSIM measures structural similarity and is fairer to sharp outputs. We therefore expect the diffusion model to look the sharpest even where its MSE is not the lowest.

---

## References

1. J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics," *ICML*, 2015.
2. J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS*, 2020.
3. J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," *ICLR*, 2021.
4. C. Saharia et al., "Image Super-Resolution via Iterative Refinement" (SR3), *IEEE TPAMI*, 2022.
5. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI*, 2015.
6. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-Based Learning Applied to Document Recognition," *Proceedings of the IEEE*, 1998.
7. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity," *IEEE TIP*, 2004.
8. Google Creative Lab, "The Quick, Draw! Dataset," https://github.com/googlecreativelab/quickdraw-dataset
9. Git, Docker and Streamlit documentation: https://git-scm.com/doc, https://docs.docker.com/, https://docs.streamlit.io/