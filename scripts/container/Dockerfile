# Adapted from kit/examples/container/Dockerfile. No original evaluation image exists:
# evaluation ran natively on Ubuntu 26.04. Copy its installed Python packages unchanged.
# Owner-only build downloads this base and the apt packages below; no pip/uv/model downloads.
ARG BASE_IMAGE=ubuntu:26.04
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates libstdc++6 libgomp1 \
    libglvnd0 libgl1 libglx0 libegl1 libgles2 libopengl0 \
    libglu1-mesa libosmesa6 libglfw3 \
    libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 \
    libxinerama1 libxfixes3 libxxf86vm1 libxkbcommon0 libxcb1 \
    fontconfig libfreetype6
COPY submission /submission
RUN chmod 755 /submission/run.sh && chmod -R a+rX /submission \
    && /submission/runtime/python-kit/bin/python3.13 -B /submission/image_setup.py \
    && /submission/runtime/kit-env/bin/python -B -c 'import numpy, pandas, pyarrow, trimesh, fast_simplification, kaggle, kagglesdk' \
    && PYTHONPATH=/submission/runtime/flashchord/src JAX_PLATFORMS=cpu \
       /submission/runtime/eval-env/bin/python -B -c 'import hydra, safetensors; from flash_chord.evaluation.harness import evaluation_code_sha256; print(evaluation_code_sha256())'
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /submission
CMD ["/submission/run.sh", "verify"]
