# Quick, Draw! Diffusion: Doodle Super-Resolution

[![tests](https://github.com/simranchikkala/dtd-diffusionproject-quickdraw/actions/workflows/test.yml/badge.svg)](https://github.com/simranchikkala/dtd-diffusionproject-quickdraw/actions/workflows/test.yml)

This project has been completed and compiled by the following students:

| Name | Student ID | Main responsibilities |
|---|---|---|
| Aditi Raj | 1012491 | App Building & Integration with Model |
| Joe | 1012340| App Building & Report |
| Rakshitha Rao HS | 1012250 | App Building & Report |
| Simran Srinivasan Chikkala | 1012380 | Model Building, Training & Docker Implementation  |

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

### How the model was improved

We developed the model over several iterations:

1. **Iteration 0 (test run):** with noise levels up to 0.5, the faint strokes of the shrunk 8×8 drawings were completely hidden by noise, and every model produced junk. We fixed this by brightening each 8×8 image so its strongest pixel is 1, and by limiting the noise level to 0.3.
2. **Iteration 1:** trained on the real Quick, Draw! data. The diffusion model drew much sharper doodles than the baselines, but often drew the wrong shape, and it scored worse than the blurry baselines on MSE and SSIM.
3. **Iteration 2:** to make the diffusion model follow its input more closely, we added **classifier-free guidance**: during training the 8×8 hint is hidden for 10% of images, so the model also learns to draw a doodle without any hint, and when generating it compares its "with hint" and "without hint" predictions and pushes the result further towards the hint. We also added a **shape accuracy** measure (a separately trained star/moon classifier that checks whether each output is the right kind of shape) and a figure showing several outputs for the same input. Guidance made the results worse: because the 8×8 hint is so small and noisy, guidance amplified noise rather than shape, and outputs filled up with scribbles.
4. **Iteration 3 (final):** we went back to generating without the extra push (guidance strength 1), while keeping the shape accuracy measure and the new figures. No retraining was needed, since the iteration 2 model at guidance 1 behaves like the iteration 1 model.

## Dataset

The dataset selected was **Google's Quick, Draw!** for the categories **star** and **moon**. Quick, Draw! contains millions of doodles drawn by people around the world in under 20 seconds each, released by Google under the CC BY 4.0 licence. We use Google's pre-made 28×28 grayscale bitmaps (white strokes on a black background) and keep a random subset of **10,000 drawings per category** (20,000 in total), split 90% for training and 10% for testing with a fixed random seed.

The raw data is not stored in this repository because the files are large (about 100 MB per category). Anyone can re-create the exact same subset by running `download_data.py`.

---

## Project structure

This project contains the following files, each with a specific function:

| File / folder | Function |
|---|---|
| `download_data.py` | Downloads the chosen Quick, Draw! categories and saves a fixed random subset to `data/quickdraw.npz`. Uses only built-in Python, so it works before other packages are installed. |
| `train.py` | Trains the linear regression, CNN, shape classifier and diffusion models and saves them. Logs the loss for each epoch to `models/train_log.csv`. |
| `evaluate.py` | Compares all four methods on the test set using MSE, SSIM and shape accuracy, tests different guidance strengths, and saves the results tables and figures to `results/`. |
| `model.pt` | The trained diffusion model used by the app. |
| `models/` | The trained baseline models (`linear.pt`, `cnn.pt`), the shape classifier (`classifier.pt`) and the training log. |
| `src/data.py` | Loading data, shrinking images, adding noise and preprocessing user images. |
| `src/models.py` | The linear regression, CNN, U-Net and shape classifier model definitions. |
| `src/diffusion.py` | The noise schedule, training loss, classifier-free guidance and step-by-step image generation. |
| `src/metrics.py` | MSE and SSIM evaluation metrics. |
| `requirements.txt` | Python packages needed to run the project. |
| `app.py` | Streamlit web app: draw, upload or pick a star or moon doodle and see the diffusion output, the denoising steps, the baselines and several samples. |
| `components/draw_canvas/` | The drawing canvas used by the app (plain HTML and JavaScript, no build step). |
| `tests/` | Automated tests (pytest), with settings in `pytest.ini`. |
| `Dockerfile` | Packages the app and its environment. |
| `.dockerignore` | Keeps the data, results and git history out of the Docker image. |
| `.github/workflows/test.yml` | Runs the tests and a Docker build automatically on every push. |

---

## Installation

The project needs **Python 3.11–3.13** (we used 3.12).

```bash
git clone https://github.com/simranchikkala/dtd-diffusionproject-quickdraw.git
cd dtd-diffusionproject-quickdraw
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
This takes about 1–1.5 hours on a laptop CPU, or a few minutes on a GPU (e.g. Google Colab). For a quick check that everything works:
```bash
python3 train.py --epochs 2 --cnn-epochs 1 --clf-epochs 1 --limit 2000
```

**3. Evaluate**
```bash
python3 evaluate.py
```
Outputs:
- `results/metrics.csv`: MSE, SSIM and shape accuracy for each method, noise level and category
- `results/guidance.csv`: diffusion scores at different guidance strengths
- `results/comparison.png`: example results side by side
- `results/guidance.png`: the same inputs generated with different guidance strengths
- `results/samples.png`: several different diffusion outputs for the same input
- `results/denoising_steps.png`: the diffusion model turning noise into a doodle

Useful options: `--guidance 1.5` changes the guidance strength; `--n 100 --steps 25 --guidance-sweep 2` gives a quicker run.

**4. Run the app**
```bash
python3 -m streamlit run app.py
```
Then open http://localhost:8501 in a browser.

## Running the tests

```bash
python3 -m pytest
```

The 38 tests in `tests/test_diffusion.py` (plus 5 end-to-end app tests in `tests/test_app.py`) run in a few seconds and don't need the dataset. They check:

- **Image sizes and noise:** 28×28 → 8×8 → 28×28, noise follows the brief's formula, the same seed gives the same noise, and the train/test split is repeatable with no overlap.
- **Models:** linear regression, CNN, U-Net and classifier all produce the expected output shapes, and saving then loading a model gives identical weights.
- **Diffusion:** the noise schedule goes from almost clean to almost pure noise, noising follows the brief's formula, training loss is finite, generated images are 28×28 with values in [0, 1] and no NaNs (with and without guidance), and the same seed gives the same image.
- **Metrics:** MSE is 0 and SSIM is 1 for identical images.
- **Preprocessing user uploads:** colour, transparent, grayscale, white-background, odd-sized, off-centre and blank images are all turned into valid 28×28 white-on-black drawings, and blank images are flagged.
- **The trained model:** `model.pt` loads and generates valid images. This test is skipped if `model.pt` hasn't been trained yet.

We checked that the tests catch real mistakes by deliberately breaking the code (for example, disabling the white-background inversion in `src/data.py`), which made the relevant tests fail.

## Docker

Docker packages the app together with the right Python version and every package it needs, so it runs the same way on any machine.

```bash
docker build -t dtd-diffusionproject-quickdraw .
docker run -p 8501:8501 dtd-diffusionproject-quickdraw
```

Then open http://localhost:8501. The image uses Python 3.12 and the CPU-only version of PyTorch to keep it small, and it includes the trained `model.pt`, so train the model before building. The first build takes a few minutes; later builds are much quicker because Docker reuses the installed packages when only the code changes.

**Windows / WSL users:** in Docker Desktop, turn on **Settings → Resources → WSL integration** for Ubuntu. If you then get a "permission denied" error for `docker.sock`, add your user to the docker group with `sudo usermod -aG docker $USER` and restart WSL with `wsl --shutdown`.

## Automated checks (GitHub Actions)

Every time someone pushes code or opens a pull request, GitHub runs two checks automatically (`.github/workflows/test.yml`):

1. **Tests:** sets up Python 3.12, installs the packages and runs all the tests.
2. **Docker build:** if the tests pass, it checks the Docker image still builds.

Each run can be followed in the repository's **Actions** tab, and the badge at the top of this page shows whether the latest run passed. A change that breaks something is caught straight away, before it reaches the working version.

## Live app

**Try it here: https://dtd-diffusionproject-quickdraw-bxqwp8vesuwxhdpw6y7t5n.streamlit.app/**

Draw a star or a moon (or upload a picture, or pick an example) and watch the diffusion model rebuild it from a noisy 8×8 version. The app also shows the denoising steps, compares the output with the bicubic, linear regression and CNN baselines, and generates several samples from the same input. It is hosted for free on Streamlit Community Cloud. If nobody has used it for a few days it goes to sleep, and the first visitor waits about 30 seconds while it wakes up.

---

## Results

Results on 300 held-out test drawings (146 stars, 154 moons), with noise level 0.1 added to the 8×8 input. Diffusion uses guidance 1 and 50 sampling steps.

| Method | MSE ↓ | SSIM ↑ | Shape accuracy ↑ |
|---|---|---|---|
| Bicubic | 0.117 | 0.208 | 0.51 |
| Linear regression | **0.064** | 0.212 | 0.51 |
| CNN | 0.066 | **0.221** | 0.52 |
| Diffusion | 0.128 | 0.170 | **0.77** |
| Real drawings (ceiling) | 0 | 1 | 0.98 |

We use three measures because each captures something different:
- **MSE** measures average pixel error. It rewards safe, blurry outputs, because a blur "hedges" between possible stroke positions.
- **SSIM** measures structural similarity. Like MSE it compares pixel by pixel, so a sharp stroke drawn slightly out of place is penalised heavily.
- **Shape accuracy** uses a separately trained star/moon classifier to check whether each output is recognisably the right kind of shape. The classifier's accuracy on the real drawings (0.98) is the best any method can reach.

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

- The linear regression and CNN baselines do best on the pixel-based metrics MSE and SSIM: their blurry output is the 'hedge' between the only two possible explanations. But they can't produce a recognisable star. 
- The diffusion model produces sharp doodles in the right shape a lot more often (77% versus around 51%), but has a higher MSE because a sharp stroke in the wrong place is punished more than a blurry one. 
- Stronger classifier-free guidance worsened every metric: the 8×8 hint is so small and noisy that guidance just adds noise. 
- Diffusion makes *a* plausible doodle, rather than *the* original: same input → different result (results/samples.png). 
- Some "incorrect" outputs aren't errors: Quick, Draw! Sketches take less than 20 seconds, and in the test set some drawings are scribbles that aren't clearly either shape.
---

## References

1. Google Creative Lab, "The Quick, Draw! Dataset," https://github.com/googlecreativelab/quickdraw-dataset
