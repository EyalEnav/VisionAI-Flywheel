# VisionAI-Flywheel 🎬

> Synthetic surveillance data pipeline — text prompt → Kimodo motion synthesis → clothed SOMA mesh render → Cosmos-Transfer2.5 Sim2Real → NVIDIA VSS annotation → VLM fine-tuning dataset

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
[![Deploy on Brev](https://brev.nvidia.com/assets/deploy-badge.svg)](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3AuTjTao5gelkXaCcUkXTRNbdyL)

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
│   ├── kimodo-api/                   # Kimodo FastAPI wrapper
│   │   ├── Dockerfile
│   │   ├── kimodo_api_server.py
│   │   └── docker-compose.yml
│   └── cosmos-transfer/              # Cosmos-Transfer2.5 API + Docker
│       ├── cosmos_api.py
│       ├── build.sh
│       └── docker-compose.yml
├── deployments/
│   └── vss/
│       ├── docker-compose.override.yml   # VSS overrides for RTX PRO 6000
│       ├── env.rtxpro6000bw              # GPU/port env profile
│       └── nginx-extra.conf              # Nginx upstream for render-api
└── helm/
    └── render-api/                   # Kubernetes Helm chart
```

---

## Cosmos Transfer2.5 parameters

Best config (confirmed): `edge=0.85 + vis=0.45`
- edge control (Canny): preserves geometry and silhouette
- vis control (blur): preserves scene structure and perspective

---


---

## 🐳 Docker Images (GHCR)

Pre-built images are available on GitHub Container Registry:

| Image | Pull Command |
|---|---|
| `render-api` | `docker pull ghcr.io/eyalenav/render-api:latest` |
| `kimodo-api` | `docker pull ghcr.io/eyalenav/kimodo-api:latest` |
| `kimodo` | `docker pull ghcr.io/eyalenav/kimodo:1.0` |

> **Note:** Images require NVIDIA GPU with CUDA 12.8+. Pull from a machine with sufficient disk space (each image is ~35–40 GB).

## License

Apache 2.0 — see [LICENSE](LICENSE)
