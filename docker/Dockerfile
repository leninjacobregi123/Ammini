# Built and run entirely via Docker -- no host installs needed on Shannon
# (no sudo there). Targets an RTX 5090 (Blackwell, sm_120), which needs a
# recent CUDA/PyTorch build; adjust the cuXXX index below if your driver is
# older (check with `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi`).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Use Ubuntu 22.04's own default python3 (3.10) rather than installing 3.11
# alongside it -- `python3-pip` is tied to whichever python3 is the OS
# default, so installing a second interpreter version and only symlinking
# `python` to it leaves pip installing into the *other* interpreter's
# site-packages (import errors at runtime despite a clean build log).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# Install torch matched to CUDA 12.8 (Blackwell/RTX 5090 support) first, as
# its own layer, so editing requirements.txt later doesn't force a torch re-download.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace
EXPOSE 8501

CMD ["bash"]
