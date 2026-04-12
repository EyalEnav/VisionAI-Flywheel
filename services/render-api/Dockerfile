FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYOPENGL_PLATFORM=egl \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for pyrender EGL + ffmpeg + OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libgles2-mesa-dev libegl1-mesa-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg git curl cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kimodo (editable install from submodule mount)
# Mounted at /kimodo via docker-compose volume
RUN echo "Kimodo will be installed at runtime from /kimodo mount"

COPY server.py .
COPY render/ render/

EXPOSE 9000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
