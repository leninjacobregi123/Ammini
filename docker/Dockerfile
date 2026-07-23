# syntax=docker/dockerfile:1
# Built and run entirely via Docker -- no host installs needed on Shannon
# (no sudo there). Based on the *official* PyTorch image rather than a bare
# CUDA base + manual `pip install torch` -- avoids re-downloading the ~5GB
# cu128 torch wheel (torch is already built in against CUDA 12.8/cuDNN9,
# which is what the RTX 5090's Blackwell/sm_120 architecture needs), and
# avoids the whole class of bug where a hand-installed second Python
# interpreter doesn't match the one `pip` actually installs for -- this
# image's python/pip are already one coherent, consistent environment.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /workspace

COPY requirements.txt .
# Shannon's link to PyPI runs ~60-130 KB/s, so a 48MB wheel (pyarrow etc.) can
# genuinely take 10+ minutes and occasionally stalls mid-download. The
# BuildKit cache mount persists pip's download cache across retries *and*
# across separate `make build` runs, so a stall doesn't mean starting over --
# already-fetched wheels are reused instead of re-downloaded from scratch.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
    || pip install -r requirements.txt \
    || pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace
EXPOSE 8501

CMD ["bash"]
