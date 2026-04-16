FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libgles2 libegl1 \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg git curl cmake build-essential

RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv 2>/dev/null; true

RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" python-multipart \
    numpy trimesh Pillow aiofiles httpx \
    "transformers>=4.45.0" accelerate

RUN pip install --no-cache-dir pyrender

RUN pip install --no-cache-dir insightface onnxruntime-gpu

RUN pip install --no-cache-dir opencv-python-headless

RUN pip install --no-cache-dir qwen-vl-utils viser peft

COPY server.py .
COPY render/ render/

EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
