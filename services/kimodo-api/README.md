# kimodo-api 🕺

> **REST API wrapper for [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo)** — text-to-motion generation as a microservice.

Kimodo is NVIDIA's kinematic motion diffusion model trained on 700 hours of commercially-licensed mocap data. This image wraps it in a simple FastAPI server so any service can generate 3D human motion from a text prompt over HTTP — no Python environment setup required.

---

## ✨ What it does

Send a text prompt → get back a `.npz` motion file (SOMA 77-joint skeleton) ready for rendering or retargeting.

```
POST /generate
{ "prompt": "person stumbles and falls after being pushed" }

→ Returns: NPZ binary (SOMA 77-joint, 30fps)
```

---

## 🐳 Quick start

```bash
docker run --rm --gpus '"device=0"' \
  -p 9551:9551 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/eyalenav/kimodo-api:latest
```

**First run** will automatically download the Kimodo-SOMA-RP-v1.1 model weights (~17 GB) from HuggingFace. Subsequent runs use the cached volume.

---

## 📋 Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 17 GB | 24 GB |
| Disk (model cache) | 20 GB | 30 GB |
| CUDA | 12.x | 12.8 |
| Driver | 535+ | 570+ |

Tested on: RTX 3090, RTX 4090, A100, RTX PRO 6000 Blackwell.

---

## 🔌 API Reference

### `GET /health`
Liveness check.
```json
{ "status": "ok" }
```

### `POST /generate`
Generate motion from text prompt.

**Body (JSON):**
```json
{
  "prompt": "person walking slowly with a limp",
  "duration": 4.0,
  "num_samples": 1,
  "seed": 42,
  "fps": 30
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Natural language motion description |
| `duration` | float | 4.0 | Motion duration in seconds |
| `num_samples` | int | 1 | Number of motion variations |
| `seed` | int | null | Random seed for reproducibility |
| `fps` | int | 30 | Output frame rate |

**Response:** `application/octet-stream` — NPZ file  
Filename in `Content-Disposition` header: `motion_<uuid>.npz`

### `GET /status`
Server info, GPU stats, loaded model name.

---

## 🔗 Use with render-api

This container is designed to pair with [`ghcr.io/eyalenav/render-api`](https://ghcr.io/eyalenav/render-api) which consumes the NPZ output and renders a clothed SOMA mesh video.

```yaml
# docker-compose.yml
services:
  kimodo-api:
    image: ghcr.io/eyalenav/kimodo-api:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    ports:
      - "9551:9551"
    volumes:
      - hf_cache:/root/.cache/huggingface

  render-api:
    image: ghcr.io/eyalenav/render-api:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - KIMODO_API_URL=http://kimodo-api:9551
    ports:
      - "9001:9000"
    volumes:
      - hf_cache:/root/.cache/huggingface

volumes:
  hf_cache:
```

---

## 🏗️ Build from source

```bash
git clone --recurse-submodules https://github.com/EyalEnav/VisionAI-Flywheel
cd VisionAI-Flywheel
docker build -t kimodo-api:local services/kimodo-api/
```

---

## 📜 License

This Docker image is released under **Apache 2.0** — see [LICENSE](LICENSE).

### Third-party components

| Component | License | Notes |
|---|---|---|
| [Kimodo](https://github.com/nv-tlabs/kimodo) | Apache 2.0 | NVIDIA nv-tlabs |
| [Kimodo model weights](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1) | NVIDIA Open Model License | Commercial use permitted |
| [SOMA-X body model](https://huggingface.co/nvidia/SOMA-X) | NVIDIA Open Model License | Commercial use permitted |
| [LLM2Vec](https://github.com/McGill-NLP/llm2vec) | MIT | Text encoder |
| [FastAPI](https://fastapi.tiangolo.com/) | MIT | — |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | — |

> **Note:** Model weights (Kimodo-SOMA-RP-v1.1, SOMA-X) are licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Commercial use is permitted. The weights are **not baked into this image** — they are downloaded from HuggingFace at runtime.

---

## 🙏 Credits

Built on top of NVIDIA's [Kimodo](https://github.com/nv-tlabs/kimodo) by the [VisionAI-Flywheel](https://github.com/EyalEnav/VisionAI-Flywheel) project.
