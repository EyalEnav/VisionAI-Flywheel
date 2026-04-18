# cosmos-transfer 🌍

> **Standalone REST API for [NVIDIA Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)** — Sim2Real video diffusion as a microservice.

Cosmos-Transfer2.5 converts synthetic/rendered videos into photorealistic footage using multi-control diffusion (edge + visual guidance). This image wraps it in an async job queue API — submit a video, poll for completion, download the result.

---

## ✨ What it does

Send a synthetic render → get back a photorealistic video.

```
POST /transfer
{ "prompt": "city street, daytime, surveillance camera", "edge": 0.85, "vis": 0.45 }
+ video file upload

→ job_id

GET /jobs/{job_id}  →  { "status": "done", "output": "/tmp/result.mp4" }
GET /video/{job_id} →  MP4 stream
```

---

## 🐳 Quick start

```bash
docker run --rm --gpus '"device=0"' \
  -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /tmp/cosmos_out:/tmp \
  ghcr.io/eyalenav/cosmos-transfer:latest
```

**First run** downloads Cosmos-Transfer2.5-2B model weights (~35 GB) from HuggingFace. Subsequent runs use the cache volume.

---

## 📋 Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 32 GB | 48 GB |
| Disk (model cache) | 40 GB | 60 GB |
| CUDA | 12.8 | 12.8 |
| Driver | 570+ | 570+ |

Tested on: A100 80GB, RTX 6000 Ada, RTX PRO 6000 Blackwell.

---

## 🔌 API Reference

### `GET /health`
```json
{ "ok": true, "queued": 0 }
```

### `POST /transfer`
Submit a Sim2Real transfer job.

**Form fields:**
| Field | Type | Default | Description |
|---|---|---|---|
| `video` | file | required | Input synthetic video (MP4) |
| `prompt` | string | required | Scene description for diffusion guidance |
| `edge` | float | 0.85 | Edge control weight (Canny, geometry preservation) |
| `vis` | float | 0.45 | Visual control weight (blur, scene structure) |
| `sigma` | float | 100.0 | Noise level — higher = more realism, less fidelity |
| `output_name` | string | auto | Output filename prefix |

**Response:**
```json
{ "job_id": "abc123", "status": "queued" }
```

### `GET /jobs/{job_id}`
Poll job status.
```json
{ "status": "running" }          // still processing
{ "status": "done", "output": "/tmp/result.mp4" }
{ "status": "error", "detail": "..." }
```

### `GET /video/{job_id}`
Stream the finished MP4 (only available when `status == "done"`).

### `GET /jobs`
List all jobs and their statuses.

---

## ⚙️ Best config (validated)

Through extensive testing across 80+ synthetic clips:

```
edge  = 0.85   # strong geometry/silhouette preservation
vis   = 0.45   # moderate scene structure guidance  
sigma = 100.0  # good realism/fidelity balance
```

Lower `sigma` → more faithful to input geometry but less photorealistic.  
Higher `sigma` → more photorealistic but may drift from input structure.

---

## 🔗 Use in the full pipeline

```yaml
# docker-compose.yml
services:
  cosmos-transfer:
    image: ghcr.io/eyalenav/cosmos-transfer:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    ports:
      - "8080:8080"
    volumes:
      - hf_cache:/root/.cache/huggingface
      - /tmp:/tmp

volumes:
  hf_cache:
```

Call from any service:
```python
import requests

with open("render.mp4", "rb") as f:
    r = requests.post("http://localhost:8080/transfer", 
        data={"prompt": "city street surveillance", "edge": 0.85, "vis": 0.45},
        files={"video": f})

job_id = r.json()["job_id"]

# Poll
while True:
    status = requests.get(f"http://localhost:8080/jobs/{job_id}").json()
    if status["status"] == "done":
        break
    time.sleep(10)

# Download
video = requests.get(f"http://localhost:8080/video/{job_id}")
open("output.mp4", "wb").write(video.content)
```

---

## 🏗️ Build from source

```bash
git clone --recurse-submodules https://github.com/EyalEnav/VisionAI-Flywheel
cd VisionAI-Flywheel
bash services/cosmos-transfer/build.sh
```

The Dockerfile uses `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` as base (publicly available, no NGC login required).

---

## 📜 License

This Docker image (API wrapper code) is released under **Apache 2.0** — see [LICENSE](LICENSE).

### Third-party components

| Component | License | Notes |
|---|---|---|
| [Cosmos-Transfer2.5 source](https://github.com/nvidia-cosmos/cosmos-transfer2.5) | Apache 2.0 | NVIDIA |
| [Cosmos-Transfer2.5-2B weights](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) | NVIDIA Open Model License | Commercial use permitted |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | — |
| [FastAPI](https://fastapi.tiangolo.com/) | MIT | — |
| CUDA base image (`nvidia/cuda:12.8.1`) | NVIDIA CUDA EULA | Publicly available from Docker Hub |

> **Note on model weights:** Cosmos-Transfer2.5-2B weights are licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) which **permits commercial use**. Weights are downloaded from HuggingFace at runtime and are **not baked into this image**.

> **Note on base image:** This image uses `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` from Docker Hub (not `nvcr.io`). No NGC account is needed to pull or use this image.

---

## 🙏 Credits

Built on top of NVIDIA's [Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5).  
Part of the [VisionAI-Flywheel](https://github.com/EyalEnav/VisionAI-Flywheel) synthetic data pipeline.
