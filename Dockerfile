# Packages the app with everything it needs, so it runs the same everywhere.
#
#   docker build -t doodle-diffusion .
#   docker run -p 8501:8501 doodle-diffusion
#   then open http://localhost:8501

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Run as a normal user rather than root (also what Hugging Face Spaces expects).
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

# Install packages first, so Docker can reuse this layer when only code changes.
# The CPU-only build of PyTorch is much smaller than the default one.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# Copy the code and the trained model.
COPY --chown=user . .
USER user

EXPOSE 8501

# Lets Docker report whether the app is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# PORT can be overridden by hosting platforms; 8501 is Streamlit's default.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]