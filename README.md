# VisionAI-Flywheel 🎬

> Synthetic surveillance data pipeline — text prompt → Kimodo motion synthesis → clothed SOMA mesh render → Cosmos-Transfer2.5 Sim2Real → NVIDIA VSS annotation → VLM fine-tuning dataset

[![kimodo-api](https://img.shields.io/badge/ghcr.io-kimodo--api-blue?logo=docker)](https://ghcr.io/eyalenav/kimodo-api)
[![cosmos-transfer](https://img.shields.io/badge/ghcr.io-cosmos--transfer-blue?logo=docker)](https://ghcr.io/eyalenav/cosmos-transfer)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Deploy on Brev](https://brev.nvidia.com/assets/deploy-badge.svg)](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3AuTjTao5gelkXaCcUkXTRNbdyL)

---

## 🐳 Published Docker Images

Two standalone microservices from this pipeline are publicly available:

### kimodo-api — Text-to-Motion REST API
```bash
docker pull ghcr.io/eyalenav/kimodo-api:latest
docker run --rm --gpus '"device=0"' -p 9551:9551 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/eyalenav/kimodo-api:latest
```
→ `POST http://localhost:9551/generate` with `{"prompt": "person walking fast"}` → NPZ motion file

### cosmos-transfer — Sim2Real Video Diffusion API
```bash
docker pull ghcr.io/eyalenav/cosmos-transfer:latest
docker run --rm --gpus '"device=0"' -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/eyalenav/cosmos-transfer:latest
```
→ `POST http://localhost:8080/transfer` with video + prompt → photorealistic MP4

See [`services/kimodo-api/README.md`](services/kimodo-api/README.md) and [`services/cosmos-transfer/README.md`](services/cosmos-transfer/README.md) for full API docs.

---

## Architecture

```
Text prompt
    │
    ▼
[kimodo-api]  ─── LLM2Vec + Llama-3-8B ──► NPZ/BVH motion  (GPU1, :9551)
    │
    ▼
[render-api]  ─── SOMA skinning + clothing colors ──► synthetic MP4  (GPU0, :9001)
    │
    ▼
[cosmos-transfer]  ─── Sim2Real diffusion (edge=0.85, vis=0.45) ──► photorealistic MP4  (GPU1, :8080)
    │
    ▼
[NVIDIA VSS]  ─── VLM annotation (Cosmos-Reason2-8B + Nemotron) ──► training pair  (GPU0, :8000)
```

---

## 🚀 Deploy on NVIDIA Brev

### One-click deploy
[![Deploy on Brev](https://brev.nvidia.com/assets/deploy-badge.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-3CPOb6lPulFg4OALeWRWYLHw24P)

### Hardware requirements
| Resource | Minimum | Recommended |
|---|---|---|
| GPU | 1× RTX 6000 Ada / A100 | 2× RTX PRO 6000 Blackwell |
| VRAM | 48 GB | 96 GB (2×48) |
| Disk | 300 GB NVMe | 500 GB NVMe |
| RAM | 64 GB | 128 GB |

### Environment variables (set in Brev before deploying)
| Variable | Description |
|---|---|
| `NGC_CLI_API_KEY` | NVIDIA NGC API key — required to pull VSS images |
| `HUGGINGFACE_TOKEN` | HuggingFace token — required for Llama-3-8B-Instruct |

---

## Services & Ports

| Service | Port | GPU | Description |
|---|---|---|---|
| `vss-agent` | 8000 | GPU0 | NVIDIA VSS video analysis (Cosmos-Reason2 + Nemotron) |
| `render-api` | 9001 | GPU0 | FastAPI: Kimodo motion → SOMA mesh render |
| `kimodo-api` | 9551 | GPU1 | Kimodo text-to-motion generation |
| `cosmos-transfer` | 8080 | GPU1 | Cosmos-Transfer2.5 Sim2Real diffusion |
| Studio UI | 9000 | — | Nginx + React UI (served at `http://<IP>:9000`) |

---

## Quick start (after setup)

```bash
# Start all services
bash scripts/start.sh

# Stop all services
bash scripts/stop.sh

# Generate a synthetic scene
curl -X POST http://localhost:9001/generate \
  -F "prompt=person pushing through a crowd, surveillance camera angle" \
  -F "clothing=blue jacket, black pants" \
  -F "use_cosmos=true"

# Check VSS
curl http://localhost:8000/health
```

---

## Project structure

```
VisionAI-Flywheel/
├── setup.sh                          # Brev first-boot setup
├── scripts/
│   ├── start.sh                      # Start all services
│   └── stop.sh                       # Stop all services
├── services/
│   ├── render-api/                   # FastAPI render server + SOMA pipeline
│   │   ├── Dockerfile
│   │   ├── server.py
│   │   └── render/soma_render.py
│   ├── kimodo-api/                   # Kimodo FastAPI wrapper 🐳 ghcr.io/eyalenav/kimodo-api
│   │   ├── README.md
│   │   ├── LICENSE
│   │   └── Dockerfile
│   └── cosmos-transfer/              # Cosmos-Transfer2.5 API 🐳 ghcr.io/eyalenav/cosmos-transfer
│       ├── README.md
│       ├── LICENSE
│       ├── build.sh
│       └── run.sh
├── deployments/
│   └── vss/
│       ├── docker-compose.override.yml
│       ├── env.rtxpro6000bw
│       └── nginx-extra.conf
└── helm/
    └── render-api/                   # Kubernetes Helm chart
```

---

## Cosmos Transfer2.5 parameters

Best config (confirmed across 80+ clips): `edge=0.85 + vis=0.45 + sigma=100`
- **edge** (Canny): preserves geometry and silhouette
- **vis** (blur): preserves scene structure and perspective
- **sigma**: noise level — 100 is the sweet spot for realism vs. fidelity

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

> NVIDIA Kimodo, SOMA-X, and Cosmos-Transfer2.5 model weights are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) (commercial use permitted). Weights are downloaded at runtime and are not included in this repository or Docker images.
