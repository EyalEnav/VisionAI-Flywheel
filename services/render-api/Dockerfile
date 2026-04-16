FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive     PYOPENGL_PLATFORM=egl     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     libgl1 libgles2 libegl1     libglib2.0-0 libsm6 libxrender1 libxext6     ffmpeg git curl cmake build-essential     && rm -rf /var/lib/apt/lists/*

# Remove conflicting cv2 from base image
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv 2>/dev/null;     find /usr/local/lib/python3.10/dist-packages/cv2 -delete 2>/dev/null; true

# Install Kimodo + all its deps directly from the mounted source
# We copy the requirements file at build time and install everything
COPY kimodo_requirements.txt .
RUN pip install --no-cache-dir -r kimodo_requirements.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY render/ render/

EXPOSE 9000
CMD uvicorn server:app --host 0.0.0.0 --port 9000 --workers 1
