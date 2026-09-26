# ---- torch-base: keep this stage IDENTICAL in Sybil/Dockerfile and CVD-Risk-Estimator/Dockerfile
# (dicom-diagnosis/scripts/__tests__/dockerfiles.test.js checks it). Built together
# (`docker compose build`), or one after the other on the same machine, BuildKit builds it once and
# both images share its layers: the ~4.9 GB of torch + CUDA libraries is stored once, not twice.
FROM python:3.10-slim AS torch-base

# pip otherwise keeps every downloaded wheel in /root/.cache/pip (1.9 GB, never used at run time).
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# System libraries the Python packages link against (found with ldd on the previous image):
# OpenCV needs GL, glib, X11 and libatomic. The full python:3.10 image and ffmpeg are not needed
# (GIFs are written by imageio through Pillow).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    libatomic1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip==24.0

# The build setup.py used to install (CUDA 12.1 wheels), pinned: 2.5.1 is the last cu121 release.
# torchaudio is not installed: nothing imports it.
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# ---- Sybil service
FROM torch-base

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
# config.py reads PYTHON_ENV, NOT ENV. This used to set ENV=prod (wrong name), so
# IS_DEV=True -> debug + reloader -> the model loaded twice (measured in P0: 2 processes, 6.5 GiB peak RAM).
ENV HOST_CONNECT=0.0.0.0 \
    PORT=5555 \
    PYTHON_ENV=prod \
    DEVICE=cuda

EXPOSE 5555

CMD ["python", "api.py"]
