FROM nvcr.io/nvidia/pytorch:25.03-py3

# pytorch:25.03 provides: torch 2.7, Python 3.12, CUDA 12.8, sm_120 (Blackwell) support
# numpy, scipy, filelock, urllib3 etc. are already present — we add only what's missing

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /usr/local/bin/cmake || true


# Copy kimodo source
COPY . /workspace/

# Step 1: packages that exist fine alongside pytorch:25.03
RUN pip install \
      "hydra-core>=1.3" \
      "omegaconf>=2.3" \
      "einops>=0.7" \
      "tqdm>=4.0" \
      "aiofiles" \
      "av>=16.1.0" \
      "trimesh>=3.21.7" \
      "boto3"

# Step 2: transformers (may pull in newer tokenizers/huggingface_hub — fine)
RUN pip install "transformers==5.1.0"

# Step 3: peft + accelerate without strict version deps (torch already in base)
RUN pip install "peft" "accelerate"

# Step 4: SOMA-X
RUN pip install "py-soma-x @ git+https://github.com/NVlabs/SOMA-X.git"

# Step 5: editable kimodo installs (no extra deps — already installed above)
RUN SKIP_MOTION_CORRECTION_IN_SETUP=1 pip install -e /workspace/MotionCorrection --no-deps && \
    SKIP_MOTION_CORRECTION_IN_SETUP=1 pip install -e /workspace --no-deps

# FastAPI server
COPY kimodo_api_server.py /workspace/server.py
RUN pip install fastapi "uvicorn[standard]"

EXPOSE 9551
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9551", "--workers", "1"]
