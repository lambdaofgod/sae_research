# SAE training container image.
#
# Build from repo root:
#   docker build -f infra/activault/pipelines/training/Dockerfile -t sae-training:latest .
#
# For minikube (load into local registry):
#   minikube image load sae-training:latest

FROM javirub/flashattention-pytorch:flashattn2.7.4-pytorch2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install uv

WORKDIR /app

# Freeze base image packages so we don't accidentally upgrade torch/flash-attn
RUN pip freeze | grep -iE '^(torch|torchvision|flash.attn|xformers)==' > /tmp/constraints.txt

# dictionary-learning is a git dep not on PyPI
RUN uv pip install --system -c /tmp/constraints.txt \
    "dictionary-learning @ git+https://github.com/saprmarks/dictionary_learning@1d2737a"

# Training deps
RUN uv pip install --system -c /tmp/constraints.txt \
    "mlflow>=3.10.1" \
    "boto3>=1.42.77" \
    "pyyaml>=6.0.3" \
    "einops>=0.7.0" \
    "jaxtyping>=0.2.0" \
    "zstandard>=0.22.0" \
    "datasets>=3.0.0" \
    "huggingface-hub>=0.25.0" \
    "fire>=0.7.0"

# Install sae_research package (no deps — they're all installed above)
COPY pyproject.toml /app/pyproject.toml
COPY sae_research/ /app/sae_research/
RUN touch /app/README.md
RUN uv pip install --system --no-deps -e .
