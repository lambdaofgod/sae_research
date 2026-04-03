FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install uv
RUN git clone https://github.com/tilde-research/activault.git /app
WORKDIR /app
RUN git checkout db6d1e4e36c2d3eb4fdce79e72be94f387eccee1
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system setuptools wheel
RUN uv pip install --system accelerate aiohttp boto3 datasets huggingface-hub matplotlib ray requests seaborn
RUN uv pip install --system --no-deps transformers==4.49.0
RUN uv pip install --system --no-build-isolation flash-attn xformers
RUN uv pip install --system --no-deps -e .
COPY datasets.json /app/pipeline/data/datasets.json

ENTRYPOINT ["activault"]
