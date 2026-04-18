# kimodo-api

REST API microservice wrapper around [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) — text-to-motion diffusion model generating 77-joint SOMA skeleton motion from natural language prompts.

---

## Installation

```bash
docker pull ghcr.io/eyalenav/kimodo-api:latest
```

### Run

```bash
docker run --rm \
  --gpus '"device=0"' \
  -p 9551:9551 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGINGFACE_TOKEN=hf_... \
  ghcr.io/eyalenav/kimodo-api:latest
```

> **First run:** downloads Llama-3-8B-Instruct (~16 GB) and Kimodo weights. Subsequent starts are fast (weights cached in `/root/.cache/huggingface`).

---

## API Reference

### `GET /health`

Check server status.

**Request**
```
GET http://localhost:9551/health
```

**Response**
```json
{
  "status": "ok"
}
```

---

### `POST /generate`

Generate a motion clip from a text prompt.

**Request**
```
POST http://localhost:9551/generate
Content-Type: application/json
```

```json
{
  "prompt": "person pushing through a crowd aggressively",
  "num_frames": 120,
  "fps": 30
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Natural language motion description |
| `num_frames` | int | `120` | Number of frames to generate |
| `fps` | int | `30` | Frames per second (metadata only) |

**Response**

Binary NPZ file (`application/octet-stream`).

The NPZ contains:
| Key | Shape | Description |
|---|---|---|
| `poses` | `(T, 77, 3)` | Joint rotations (axis-angle) per frame |
| `trans` | `(T, 3)` | Root translation per frame |
| `betas` | `(16,)` | SMPL body shape parameters |

**Example**
```bash
curl -X POST http://localhost:9551/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "person falling to the ground after being pushed"}' \
  --output output_motion.npz
```

---

### `POST /generate_bvh`

Generate motion and return as BVH (Biovision Hierarchy) format.

**Request**
```
POST http://localhost:9551/generate_bvh
Content-Type: application/json
```

```json
{
  "prompt": "two people fighting, punches thrown",
  "num_frames": 150
}
```

**Response**

BVH text file (`text/plain`).

**Example**
```bash
curl -X POST http://localhost:9551/generate_bvh \
  -H "Content-Type: application/json" \
  -d '{"prompt": "drunk person stumbling and falling"}' \
  --output output_motion.bvh
```

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3090 (24 GB VRAM) | RTX 6000 Ada / A100 |
| VRAM | 24 GB | 48 GB |
| RAM | 32 GB | 64 GB |
| Disk | 50 GB | 100 GB |
| CUDA | 12.1+ | 12.8 |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HUGGINGFACE_TOKEN` | Yes | HF token with access to `meta-llama/Meta-Llama-3-8B-Instruct` |
| `CUDA_VISIBLE_DEVICES` | No | Limit to specific GPU (e.g. `"0"`) |
| `PORT` | No | Override default port `9551` |

---

## Integration with VisionAI-Flywheel

`kimodo-api` is designed to run alongside `render-api` and `cosmos-transfer` as part of the full pipeline:

```yaml
# docker-compose.yml excerpt
services:
  kimodo-api:
    image: ghcr.io/eyalenav/kimodo-api:latest
    ports:
      - "9551:9551"
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
import numpy as np
import io

def generate_motion(prompt: str, num_frames: int = 120) -> dict:
    """Generate motion NPZ from text prompt."""
    response = requests.post(
        "http://localhost:9551/generate",
        json={"prompt": prompt, "num_frames": num_frames},
        timeout=120
    )
    response.raise_for_status()
    
    npz = np.load(io.BytesIO(response.content))
    return {
        "poses": npz["poses"],   # (T, 77, 3)
        "trans": npz["trans"],   # (T, 3)
        "betas": npz["betas"],   # (16,)
    }

# Example usage
motion = generate_motion("security guard running toward an incident")
print(f"Generated {motion['poses'].shape[0]} frames")
```

---

## License

Apache 2.0

> Kimodo model weights are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Weights are downloaded at runtime and are not bundled in this image.
