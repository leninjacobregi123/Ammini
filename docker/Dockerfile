# Built and run entirely via Docker -- no host installs needed on Shannon
# (no sudo there). Targets an RTX 5090 (Blackwell, sm_120), which needs a
# recent CUDA/PyTorch build; adjust the cuXXX index below if your driver is
# older (check with `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi`).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

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
