# VisionAI-Flywheel 🎬

> Synthetic surveillance data pipeline — Kimodo motion synthesis → clothed mesh render → Cosmos-Transfer2.5 Sim2Real → NVIDIA VSS annotation

## Architecture

```
Text prompt
    │
    ▼
[Kimodo API]  ─── LLM2Vec + Llama-3-8B ──► NPZ/BVH motion
    │
    ▼
[render-api]  ─── SOMA skinning + clothing ──► synthetic MP4
    │
    ▼
[Cosmos-Transfer2.5]  ─── Sim2Real diffusion ──► photorealistic MP4
    │
    ▼
[NVIDIA VSS]  ─── VLM annotation ──► fine-tuning dataset
```

---

## 🚀 Deploy on NVIDIA Brev

### One-click deploy
[![Deploy on Brev](https://brev.nvidia.com/assets/deploy-badge.svg)](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3AuTjTao5gelkXaCcUkXTRNbdyL)

### Requirements
| Resource | Minimum | Recommended |
|---|---|---|
| GPU | 1× RTX 6000 Ada / A100 | 2× RTX PRO 6000 Blackwell |
| VRAM | 48 GB | 96 GB (2×48) |
| Disk | 300 GB | 500 GB |
| RAM | 64 GB | 128 GB |

### Environment Variables (set in Brev before deploying)
| Variable | Description |
|---|---|
| `NGC_CLI_API_KEY` | NVIDIA NGC API key — [get it here](https://org.ngc.nvidia.com/setup) |
| `HUGGINGFACE_TOKEN` | HuggingFace token for Llama-3-8B — [get it here](https://huggingface.co/settings/tokens) |

### Brev setup script
The Launchable uses `setup.sh` in the repo root. It:
1. Configures Docker storage on the large disk
2. Clones this repo + submodules
3. Logs into NGC Docker registry
4. Builds `render-api`, `kimodo-api`, `cosmos-transfer` images
5. Pulls all VSS NIM images (~10 min)
6. Installs a systemd service for **auto-start on every reboot**
7. Starts the full stack

---

## Services & Ports

| Service | Port | Description |
|---|---|---|
| VSS Agent (chat/query) | 8000 | VLM video Q&A |
| VST (video upload) | 30888 | Video ingest |
| render-api | 9000 | Motion → video pipeline |
| Studio UI | 3000 | Web dashboard |

**Brev tunnel URLs** (created automatically):
- `https://8000-<instance>.brevlab.com` — VSS Agent
- `https://9000-<instance>.brevlab.com` — render-api
- `https://3000-<instance>.brevlab.com` — Studio UI

---

## Manual Usage

### Start stack
```bash
bash scripts/start.sh
```

### Stop stack
```bash
bash scripts/stop.sh
```

### Update to latest code + restart
```bash
bash scripts/update.sh
```

### Check status
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
curl http://localhost:8000/health
curl http://localhost:9000/health
```

---

## Pipeline API

### Generate synthetic video
```bash
curl -X POST http://localhost:9000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "person running through crowd",
    "clothing": "red jacket, blue jeans",
    "texture_mode": "clothed",
    "cosmos_sim2real": true
  }'
```

### Upload video to VSS & query
```bash
# Upload
curl -X POST http://localhost:8000/api/v1/videos \
  -H "Content-Type: application/json" \
  -d '{"filename": "myvideo.mp4"}'

# Query  
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "describe video myvideo.mp4"}]}'
```

---

## Repository Structure

```
VisionAI-Flywheel/
├── setup.sh                      ← Brev Launchable entry point
├── scripts/
│   ├── start.sh                  ← Start all services
│   ├── stop.sh                   ← Stop all services
│   └── update.sh                 ← git pull + restart
├── deployments/
│   └── vss/
│       ├── docker-compose.override.yml
│       ├── env.rtxpro6000bw      ← GPU/port config
│       └── nginx-extra.conf
├── services/
│   ├── render-api/               ← SOMA mesh rendering
│   ├── kimodo-api/               ← Motion synthesis
│   └── cosmos-transfer/          ← Sim2Real diffusion
├── video-search-and-summarization/  ← VSS submodule
└── helm/                         ← Kubernetes charts
```

---

## License
Apache 2.0 — see [LICENSE](LICENSE)
