"""
Render API — FastAPI server
Endpoints:
  GET  /health          → liveness
  GET  /status          → GPU info, loaded models, available motions
  POST /generate        → start async render job
  GET  /jobs/{job_id}   → poll job status + progress
  GET  /render/video/{job_id} → stream finished MP4
"""
import os, sys, uuid, asyncio, subprocess, json, glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Kimodo is installed from the mounted submodule
sys.path.insert(0, "/kimodo")

app = FastAPI(title="VLM Render API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Job store ────────────────────────────────────────────────────────────────
JOBS: dict[str, dict] = {}
KIMODO_OUTPUT_DIR = Path(os.environ.get("KIMODO_OUTPUT_DIR", "/kimodo_output"))
RENDER_OUTPUT_DIR = Path(os.environ.get("RENDER_OUTPUT_DIR", "/render_output"))
RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COSMOS_REASON2_URL   = os.environ.get("COSMOS_REASON2_URL",   "http://localhost:30082")
COSMOS_TRANSFER_URL  = os.environ.get("COSMOS_TRANSFER_URL",  "http://cosmos-transfer:8080")
INSIGHTFACE_MODEL  = os.environ.get("INSIGHTFACE_MODEL", "/models/inswapper_128.onnx")

# ─── GPU status ───────────────────────────────────────────────────────────────
def get_gpu_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]).decode()
        gpus = []
        for line in out.strip().splitlines():
            idx, name, mu, mt, util = [x.strip() for x in line.split(",")]
            gpus.append({"index": idx, "name": name,
                         "mem_used": f"{mu} MiB", "mem_total": f"{mt} MiB",
                         "util": f"{util}%"})
        return gpus
    except Exception:
        return []


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    motions = [p.name for p in KIMODO_OUTPUT_DIR.glob("*.npz")]
    return {
        "ok": True,
        "gpus": get_gpu_info(),
        "motions_available": motions,
        "jobs_active": sum(1 for j in JOBS.values() if j["status"] == "running"),
        "models": {"huggingface": ["meta-llama/Meta-Llama-3-8B-Instruct"]}
    }


# ─── Generate endpoint ────────────────────────────────────────────────────────
@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    texture_mode: str = Form("clothed"),   # clothed | skeleton | faceswap | transfer
    cosmos_prompt: str = Form(""),
    motion_file: str = Form(""),          # existing .npz filename, or "" to generate
    fps: int = Form(30),
    width: int = Form(640),
    height: int = Form(480),
    face_image: Optional[UploadFile] = File(None),
    cosmos_transfer: str = Form("false"),   # "true" to run Sim2Real after render
):
    job_id = str(uuid.uuid4())
    face_path = None

    if face_image and texture_mode == "faceswap":
        face_path = str(RENDER_OUTPUT_DIR / f"{job_id}_face.jpg")
        with open(face_path, "wb") as f:
            f.write(await face_image.read())

    JOBS[job_id] = {
        "status": "queued", "progress": 0,
        "log": [], "cosmos_response": None, "error": None
    }

    npz_path = str(KIMODO_OUTPUT_DIR / motion_file) if motion_file else None

    do_cosmos = cosmos_transfer.lower() in ("true", "1", "yes")
    background_tasks.add_task(
        run_render_job, job_id, prompt,
        cosmos_prompt or prompt,   # use unified prompt for texture if no separate one
        texture_mode, npz_path, face_path, fps, width, height, do_cosmos
    )
    return {"job_id": job_id}


async def run_render_job(job_id, prompt, texture_prompt, texture_mode,
                          npz_path, face_path, fps, W, H, do_cosmos=False):
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        job["log"].append("Starting render pipeline...")

        # 1. Generate motion if needed
        if not npz_path or not Path(npz_path).exists():
            job["log"].append(f"Generating motion from prompt: {prompt}")
            job["progress"] = 5
            npz_path = await generate_kimodo_motion(job_id, prompt)
            job["log"].append(f"Motion generated: {npz_path}")
        else:
            job["log"].append(f"Using existing motion: {Path(npz_path).name}")

        job["progress"] = 20

        # 2. Get texture colors from prompt (no external NIM needed)
        colors = None
        if texture_mode in ("clothed", "cosmos"):
            job["log"].append("Parsing texture colors from prompt...")
            colors = parse_cosmos_colors(texture_prompt)
            job["log"].append("Texture colors set")
        elif texture_mode == "skeleton":
            job["log"].append("Skeleton-only mode (Transfer2.5 will add texture)")

        job["progress"] = 35

        # 3. Render SOMA mesh
        out_video = str(RENDER_OUTPUT_DIR / f"{job_id}.mp4")
        effective_mode = "cosmos" if texture_mode in ("clothed", "cosmos", "transfer") else texture_mode
        job["log"].append(f"Rendering SOMA mesh ({effective_mode} mode)...")
        await asyncio.get_event_loop().run_in_executor(
            None, render_soma_video,
            npz_path, out_video, effective_mode, colors, face_path, fps, W, H
        )

        # 4. Optional: Cosmos Transfer2.5 Sim2Real
        if do_cosmos or texture_mode == "transfer":
            job["log"].append("Running Cosmos Transfer2.5 Sim2Real...")
            job["progress"] = 70
            out_video = await run_cosmos_transfer(job_id, out_video, texture_prompt)
            job["log"].append(f"Cosmos Transfer done → {out_video}")

        job["progress"] = 100
        job["status"]   = "done"
        job["log"].append(f"Done → {out_video}")

    except Exception as e:
        job["status"] = "error"
        job["error"]  = str(e)
        job["log"].append(f"ERROR: {e}")
        import traceback; traceback.print_exc()


KIMODO_API_URL = os.environ.get("KIMODO_API_URL", "http://kimodo-api:9551")

async def generate_kimodo_motion(job_id, prompt):
    """Call kimodo-api HTTP service (pytorch:25.03 container with sm_120 support)."""
    import aiohttp
    out_npz = str(RENDER_OUTPUT_DIR / f"{job_id}_motion.npz")
    filename = f"{job_id}_motion.npz"
    payload = {"prompt": prompt, "output_filename": filename}
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{KIMODO_API_URL}/generate", json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"kimodo-api HTTP {resp.status}: {body[:300]}")
            data = await resp.json()
    # kimodo-api writes to /kimodo_output volume; we must read from KIMODO_OUTPUT_DIR
    import time
    kimodo_path = KIMODO_OUTPUT_DIR / filename
    for _ in range(20):
        if kimodo_path.exists():
            break
        time.sleep(0.5)
    if not kimodo_path.exists():
        raise RuntimeError(f'Motion file not found in kimodo_output: {filename}')
    return str(kimodo_path)


async def run_cosmos_transfer(job_id: str, input_video: str, prompt: str) -> str:
    """Call the cosmos-transfer container API for Sim2Real conversion."""
    import httpx
    output_video = str(RENDER_OUTPUT_DIR / f"{job_id}_cosmos.mp4")
    payload = {
        "input_path":  input_video,
        "output_path": output_video,
        "prompt":      prompt,
        "control_weight": 1.0
    }
    async with httpx.AsyncClient(timeout=1800) as client:
        # Start job
        resp = await client.post(f"{COSMOS_TRANSFER_URL}/transfer", json=payload)
        resp.raise_for_status()
        transfer_job_id = resp.json()["job_id"]
        # Poll until done
        import asyncio as _aio
        for _ in range(600):
            await _aio.sleep(3)
            poll = await client.get(f"{COSMOS_TRANSFER_URL}/jobs/{transfer_job_id}")
            state = poll.json()
            if state["status"] == "done":
                return output_video
            if state["status"] == "error":
                raise RuntimeError("Cosmos Transfer failed: " + str(state.get("error")))
        raise TimeoutError("Cosmos Transfer timed out after 1800s")


async def get_cosmos_texture(prompt):
    """Query Cosmos-Reason2-8B NIM for RGB clothing colors."""
    import httpx
    payload = {
        "model": "nvidia/cosmos-reason2-8b",
        "messages": [{"role": "user", "content": f"""You are a 3D texture artist.
Clothing description: "{prompt}"

Output ONLY 7 lines:
Torso RGB: R, G, B
Legs RGB: R, G, B
Shoes RGB: R, G, B
Skin RGB: 185, 140, 100
Hair RGB: 30, 20, 15
Belt RGB: R, G, B
Socks RGB: R, G, B

Rules: Skin = realistic human tone. Hair = dark unless specified. Match description exactly."""}],
        "max_tokens": 200,
        "temperature": 0.3
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{COSMOS_REASON2_URL}/v1/chat/completions", json=payload)
        data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    colors = parse_cosmos_colors(raw)
    return colors, raw


def parse_cosmos_colors(text):
    import re
    zones = {"Torso": [90,105,120], "Legs": [28,38,70], "Shoes": [35,30,25],
             "Skin": [195,155,120], "Hair": [30,20,15], "Belt": [20,15,10], "Socks": [220,220,220]}
    colors = {}
    for zone, default in zones.items():
        m = re.search(rf'{zone}[^\n]*?(\d+)[,\s]+(\d+)[,\s]+(\d+)', text, re.IGNORECASE)
        if m:
            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            colors[zone] = np.array([r, g, b, 255], dtype=np.uint8)
        else:
            colors[zone] = np.array(default + [255], dtype=np.uint8)
    return colors


def render_soma_video(npz_path, out_video, texture_mode, colors, face_path, fps, W, H):
    """Render SOMA skeleton → MP4. Imported from render/soma_render.py"""
    from render.soma_render import render
    render(npz_path=npz_path, out_video=out_video, texture_mode=texture_mode,
           colors=colors, face_path=face_path, fps=fps, W=W, H=H)


# ─── Job polling & video serving ─────────────────────────────────────────────
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JOBS[job_id]


@app.get("/render/video/{job_id}")
def get_video(job_id: str):
    path = RENDER_OUTPUT_DIR / f"{job_id}.mp4"
    if not path.exists():
        return JSONResponse({"error": "video not ready"}, status_code=404)
    return FileResponse(str(path), media_type="video/mp4",
                        filename=f"render_{job_id[:8]}.mp4")


# ─── Docker status ────────────────────────────────────────────────────────────
import subprocess, json as _json

WATCHED = ['render-api', 'vss-agent', 'vst', 'cosmos-transfer', 'nginx']

@app.get('/render/docker-ps')
def docker_ps():
    try:
        out = subprocess.check_output(
            ['docker', 'ps', '-a', '--format', '{{json .}}'],
            stderr=subprocess.STDOUT, timeout=10
        ).decode()
        rows = [_json.loads(l) for l in out.strip().splitlines() if l]
        containers = []
        for name in WATCHED:
            match = next((r for r in rows if name in r.get('Names', '')), None)
            if match:
                containers.append({
                    'name': name,
                    'status': match.get('Status', ''),
                    'state': 'running' if 'Up' in match.get('Status','') else 'stopped',
                    'image': match.get('Image', ''),
                })
            else:
                containers.append({'name': name, 'status': 'not found', 'state': 'missing', 'image': ''})
        return {'containers': containers}
    except Exception as e:
        return {'error': str(e), 'containers': []}


# ─── Service control (start / stop individual services) ──────────────────────
import shlex

SERVICES = {
    'vss': {
        'dir': '/home/ubuntu/video-search-and-summarization/deployments',
        'up':  'docker compose -f compose.yml -f /home/ubuntu/vlm-pipeline/deployments/vss/docker-compose.override.yml --env-file /home/ubuntu/vlm-pipeline/deployments/vss/env.rtxpro6000bw up -d',
        'down':'docker compose -f compose.yml -f /home/ubuntu/vlm-pipeline/deployments/vss/docker-compose.override.yml down',
    },
    'kimodo': {
        'dir': '/home/ubuntu/kimodo',
        'up':  'docker compose up -d',
        'down':'docker compose down',
    },
    'render-api': {
        'dir': '/home/ubuntu/vlm-pipeline/services/render-api',
        'up':  'docker compose up -d',
        'down':'docker compose down',
    },
    'cosmos-transfer': {
        'dir': '/home/ubuntu/vlm-pipeline/services/cosmos-transfer',
        'up':  'docker compose up -d',
        'down':'docker compose down',
    },
    'nginx': {
        'dir': '/home/ubuntu/vlm-pipeline/deployments/vss',
        'up':  'docker compose up -d nginx',
        'down':'docker compose stop nginx',
    },
}

@app.post('/render/service/{service}/{action}')
def service_control(service: str, action: str):
    if service not in SERVICES:
        return JSONResponse({'error': f'unknown service: {service}'}, status_code=400)
    if action not in ('start', 'stop'):
        return JSONResponse({'error': 'action must be start or stop'}, status_code=400)
    cfg = SERVICES[service]
    cmd = cfg['up'] if action == 'start' else cfg['down']
    try:
        # Run docker compose on the HOST filesystem via the mounted docker socket
        full_cmd = f'cd {cfg[dir]} && {cmd}'
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True, timeout=120,
            env={**__import__('os').environ, 'HOME': '/root', 'PATH': '/usr/bin:/usr/local/bin:/bin'}
        )
        return {
            'service': service, 'action': action,
            'returncode': result.returncode,
            'stdout': result.stdout[-500:],
            'stderr': result.stderr[-500:],
        }
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


@app.post('/render/startup')
def full_startup():
    """Run startup.sh on the host via nsenter"""
    try:
        result = subprocess.run(
            'bash /home/ubuntu/vlm-pipeline/scripts/startup.sh',
            shell=True, capture_output=True, text=True, timeout=300,
            env={**__import__('os').environ, 'HOME': '/root', 'PATH': '/usr/bin:/usr/local/bin:/bin'}
        )
        return {'returncode': result.returncode, 'stdout': result.stdout[-1000:], 'stderr': result.stderr[-500:]}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)




# ─── Motion list ─────────────────────────────────────────────────────────────
@app.get('/motion/list')
def motion_list():
    """List available NPZ motion files in the Kimodo output directory."""
    try:
        files = sorted([f.name for f in KIMODO_OUTPUT_DIR.glob('*.npz')])
        return {'files': files}
    except Exception as e:
        return {'files': [], 'error': str(e)}

# ─── Motion preview ──────────────────────────────────────────────────────────
@app.get('/motion/preview/{filename}')
def motion_preview(filename: str, max_frames: int = 60, step: int = 2):
    """
    Return a downsampled skeleton trajectory for preview.
    posed_joints: [frames, 77, 3] → return every  frames up to max_frames.
    Also returns root_positions for trajectory overlay.
    """
    import numpy as np
    # Sanitize filename
    safe = Path(filename).name
    npz_path = KIMODO_OUTPUT_DIR / safe
    if not npz_path.exists():
        return JSONResponse({'error': f'{safe} not found'}, status_code=404)
    try:
        d = np.load(str(npz_path), allow_pickle=True)
        joints = d['posed_joints']          # [F, 77, 3]
        roots  = d['root_positions']        # [F, 3]
        total_frames = joints.shape[0]
        # Downsample
        idx = list(range(0, min(total_frames, max_frames * step), step))
        joints_sub = joints[idx]            # [N, 77, 3]
        roots_sub  = roots[idx]             # [N, 3]
        # Normalize to 0-1 range for canvas rendering
        mn, mx = joints_sub.min(), joints_sub.max()
        joints_norm = ((joints_sub - mn) / (mx - mn + 1e-8)).tolist()
        # Root trajectory (XZ plane)
        rmin, rmax = roots_sub[:, [0,2]].min(), roots_sub[:, [0,2]].max()
        traj = ((roots_sub[:, [0,2]] - rmin) / (rmax - rmin + 1e-8)).tolist()
        return {
            'filename': safe,
            'total_frames': total_frames,
            'preview_frames': len(idx),
            'fps': 30,
            'joints': joints_norm,    # [N, 77, 3] normalized
            'trajectory': traj,       # [N, 2] XZ plane
        }
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

# ─── Kimodo generate (direct, for Motion Library UI) ─────────────────────────
class KimodoGenerateRequest(BaseModel):
    prompt: str
    output_filename: str | None = None
    num_frames: int = 150

@app.post("/kimodo/generate")
async def kimodo_generate_direct(req: KimodoGenerateRequest):
    """Proxy to kimodo-api /generate — used by Motion Library UI."""
    import aiohttp
    filename = req.output_filename or f"clip_{uuid.uuid4().hex[:8]}.npz"
    payload  = {"prompt": req.prompt, "output_filename": filename, "num_frames": req.num_frames}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{KIMODO_API_URL}/generate", json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                body = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(f"kimodo-api {resp.status}: {body}")
        return {"ok": True, "filename": filename, "output_path": f"/kimodo_output/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
