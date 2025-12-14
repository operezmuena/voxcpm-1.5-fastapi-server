FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONUNBUFFERED=1 \
    TORCHDYNAMO_DISABLE=1

# Pin VoxCPM repo + commit for reproducibility
ARG VOXCPM_REPO=https://github.com/OpenBMB/VoxCPM.git
ARG VOXCPM_COMMIT=aabda60833e0eeb413ef4d2434da82b6bff290ff

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# --- Clone VoxCPM at a fixed commit (instead of COPY VoxCPM/) ---
RUN git clone ${VOXCPM_REPO} /workspace/VoxCPM && \
    cd /workspace/VoxCPM && \
    git checkout ${VOXCPM_COMMIT}

# --- Copy your API + wrapper (kept at /workspace so uvicorn can import marcus_api:app) ---
# NOTE: Your runtime docker run uses --env-file on the host, so copying the env file into the
# image is optional. Keeping it here is fine if it contains no secrets.
COPY marcus_voxcpm.env /workspace/marcus_voxcpm.env
COPY marcus_api.py /workspace/marcus_api.py
COPY marcus_voxcpm.py /workspace/marcus_voxcpm.py
COPY README.md /workspace/README.md

# --- Copy reference audio/text ---
COPY ref/ /workspace/ref/

WORKDIR /workspace/VoxCPM

# 1) Model + HF stack (NO deps so we don't touch torch/torchvision)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-deps \
      "transformers==4.57.3" \
      "huggingface-hub>=0.34.0,<1.0" \
      "datasets>=2,<4" \
      "torchaudio==2.9.1" \
      addict \
      argbind \
      einops \
      funasr \
      gradio \
      inflect \
      modelscope \
      pydantic \
      safetensors \
      simplejson \
      sortedcontainers \
      soundfile \
      spaces \
      tqdm \
      wetext \
      "nvidia-modelopt"

# 2) Web / API stack (python-multipart REQUIRED for UploadFile/form-data)
RUN pip install --no-cache-dir \
      fastapi \
      "uvicorn[standard]" \
      requests \
      python-multipart

# 3) Install VoxCPM from source (pinned commit)
RUN pip install --no-cache-dir -e . --no-deps

# Runtime dirs (you'll typically bind-mount these)
RUN mkdir -p /workspace/episodes

EXPOSE 8000

WORKDIR /workspace
CMD ["uvicorn", "--app-dir", "/workspace", "marcus_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

