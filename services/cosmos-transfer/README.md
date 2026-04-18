# cosmos-transfer

REST API microservice wrapper around [NVIDIA Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) — a video diffusion model that converts synthetic renders into photorealistic video (Sim2Real).

---

## Installation

```bash
docker pull ghcr.io/eyalenav/cosmos-transfer:latest
```

### Run

```bash
docker run --rm \
  --gpus '"device=0"' \
  -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGINGFACE_TOKEN=hf_... \
  ghcr.io/eyalenav/cosmos-transfer:latest
```

> **First run:** downloads Cosmos-Transfer2.5-2B weights (~20 GB). Subsequent starts are fast.

---

## API Reference

### `GET /health`

Check server status.

**Request**
```
GET http://localhost:8080/health
```

**Response**
```json
{
  "status": "ok",
  "model": "Cosmos-Transfer2.5-2B",
  "device": "cuda:0"
}
```

---

### `POST /transfer`

Convert a synthetic video to photorealistic using multicontrol (edge + visual).

**Request**
```
POST http://localhost:8080/transfer
Content-Type: multipart/form-data
```

| Field | Type | Default | Description |
|---|---|---|---|
| `video` | file | required | Input synthetic MP4 (max 10s @ 24fps recommended) |
| `prompt` | string | `""` | Text describing the scene (improves realism) |
| `edge_strength` | float | `0.85` | Canny edge control strength (geometry preservation) |
| `vis_strength` | float | `0.45` | Visual/blur control strength (scene structure) |
| `sigma` | int | `100` | Noise level — lower = more faithful, higher = more realistic |
| `num_steps` | int | `35` | Diffusion steps (more = slower but higher quality) |
| `seed` | int | `-1` | Random seed (`-1` = random) |

**Response**

Binary MP4 file (`video/mp4`).

**Example**
```bash
curl -X POST http://localhost:8080/transfer \
  -F "video=@synthetic_render.mp4" \
  -F "prompt=surveillance camera footage of a crowded urban street, overcast day" \
  -F "edge_strength=0.85" \
  -F "vis_strength=0.45" \
  -F "sigma=100" \
  --output photorealistic.mp4
```

---

### `POST /transfer_async`

Submit a job and poll for completion (recommended for long clips).

**Submit**
```bash
curl -X POST http://localhost:8080/transfer_async \
  -F "video=@render.mp4" \
  -F "prompt=security incident, parking lot" \
  -F "edge_strength=0.85" \
  --output job.json
# {"job_id": "abc123", "status": "queued"}
```

**Poll**
```bash
curl http://localhost:8080/status/abc123
# {"job_id": "abc123", "status": "running", "progress": 0.42}
# ...
# {"job_id": "abc123", "status": "done"}
```

**Download**
```bash
curl http://localhost:8080/result/abc123 --output photorealistic.mp4
```

---

## Tuned Parameters

Tested across 80+ surveillance clips — confirmed sweet spot:

```
edge_strength=0.85 + vis_strength=0.45 + sigma=100
```

| Parameter | Value | Effect |
|---|---|---|
| `edge_strength` | **0.85** | Strong silhouette/geometry preservation from Canny edges |
| `vis_strength` | **0.45** | Moderate scene structure via visual blur control |
| `sigma` | **100** | Balanced noise — realistic textures without losing layout |

### When to adjust

| Scenario | Adjustment |
|---|---|
| Subject drifts from synthetic pose | Increase `edge_strength` → 0.90–0.95 |
| Background too synthetic-looking | Increase `vis_strength` → 0.55–0.65 |
| Output too faithful to render colors | Increase `sigma` → 120 |
| Too much motion blur | Decrease `sigma` → 80 |

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU | A100 40GB / RTX 6000 Ada | H100 / RTX PRO 6000 Blackwell |
| VRAM | 40 GB | 48+ GB |
| RAM | 64 GB | 128 GB |
| Disk | 30 GB | 50 GB |
| CUDA | 12.1+ | 12.8 |

**Processing time (RTX PRO 6000 Blackwell, 96GB VRAM):**
- 4s clip @ 24fps → ~3 min
- 10s clip @ 24fps → ~7 min

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HUGGINGFACE_TOKEN` | Yes | HF token with access to `nvidia/Cosmos-Transfer2.5-2B` |
| `CUDA_VISIBLE_DEVICES` | No | Limit to specific GPU (e.g. `"1"`) |
| `PORT` | No | Override default port `8080` |

---

## Integration with VisionAI-Flywheel

```yaml
# docker-compose.yml excerpt
services:
  cosmos-transfer:
    image: ghcr.io/eyalenav/cosmos-transfer:latest
    ports:
      - "8080:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    volumes:
      - hf_cache:/root/.cache/huggingface
    environment:
      - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
```

Full `docker-compose.yml`: [github.com/EyalEnav/VisionAI-Flywheel](https://github.com/EyalEnav/VisionAI-Flywheel)

---

## Example: Full Python client

```python
import requests
import time

def transfer_video(
    input_path: str,
    output_path: str,
    prompt: str = "",
    edge_strength: float = 0.85,
    vis_strength: float = 0.45,
    sigma: int = 100
):
    """Convert synthetic video to photorealistic."""
    with open(input_path, "rb") as f:
        response = requests.post(
            "http://localhost:8080/transfer",
            files={"video": ("input.mp4", f, "video/mp4")},
            data={
                "prompt": prompt,
                "edge_strength": edge_strength,
                "vis_strength": vis_strength,
                "sigma": sigma,
            },
            timeout=600
        )
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Saved to {output_path}")

# Example usage
transfer_video(
    input_path="soma_render.mp4",
    output_path="photorealistic.mp4",
    prompt="surveillance camera, urban street, daytime, overcast sky"
)
```

---

## License

Apache 2.0

> Cosmos-Transfer2.5 model weights are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Weights are downloaded at runtime and are not bundled in this image.
