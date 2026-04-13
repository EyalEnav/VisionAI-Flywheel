FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /usr/local/bin/cmake || true

# Copy kimodo source
COPY . /workspace/

# Patch docker_requirements.txt — remove kimodo-viser (optional UI, not needed for generation)
RUN sed -i '/-e .\/kimodo-viser/d' /workspace/docker_requirements.txt

# Install deps (editable install of kimodo + MotionCorrection)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    SKIP_MOTION_CORRECTION_IN_SETUP=1 pip install -r /workspace/docker_requirements.txt

# FastAPI server for generation endpoint
COPY kimodo_api_server.py /app/server.py
RUN pip install fastapi uvicorn[standard]

EXPOSE 9551

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9551"]
