"""
Render API — FastAPI server
Endpoints:
  GET  /health                → liveness
  GET  /status                → GPU info, loaded models, available motions
  POST /generate              → start async render job
  GET  /jobs/{job_id}         → poll job status + progress
  GET  /render/video/{job_id} → stream finished MP4
  POST /analyze               → analyze a video file with a VLM (pluggable backend)
  POST /generate_and_analyze  → render + analyze in one call
"""
import os, sys, uuid, asyncio, subprocess, json, glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Kimodo is installed from the mounted submodule
sys.path.insert(0, "/kimodo")

app = FastAPI(title="VLM Render API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Job store ────────────────────────────────────────────────────────────────
JOBS: dict[str, dict] = {}
KIMODO_OUTPUT_DIR = Path(os.environ.get("KIMODO_OUTPUT_DIR", "/kimodo_output"))
RENDER_OUTPUT_DIR = Path(os.environ.get("RENDER_OUTPUT_DIR", "/tmp/render_output"))
RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COSMOS_REASON2_URL = os.environ.get("COSMOS_REASON2_URL", "http://localhost:30082")
INSIGHTFACE_MODEL  = os.environ.get("INSIGHTFACE_MODEL", "/models/inswapper_128.onnx")

# Default VLM backend — override with VLM_BACKEND env var
# Options: vss | qwen | qwen7b | openai | nim
VLM_BACKEND = os.environ.get("VLM_BACKEND", "vss")

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
async def status():
    from render.vlm_analyze import vllm_health, AVAILABLE_BACKENDS
    motions = [p.name for p in KIMODO_OUTPUT_DIR.glob("*.npz")]
    vllm_info = await vllm_health()
    return {
        "ok": True,
        "gpus": get_gpu_info(),
        "motions_available": motions,
        "jobs_active": sum(1 for j in JOBS.values() if j["status"] == "running"),
        "vlm_backend": VLM_BACKEND,
        "vlm_backends_available": AVAILABLE_BACKENDS,
        "vllm": vllm_info,
    }


@app.get("/vllm/health")
async def vllm_health_endpoint():
    from render.vlm_analyze import vllm_health
    return await vllm_health()


@app.get("/vllm/models")
async def vllm_models():
    from render.vlm_analyze import list_vllm_models
    models = await list_vllm_models()
    return {"models": models}


# ─── Generate endpoint ────────────────────────────────────────────────────────
@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    texture_mode: str = Form("cosmos"),   # cosmos | skeleton | faceswap
    cosmos_prompt: str = Form(""),
    motion_file: str = Form(""),          # existing .npz filename, or "" to generate
    fps: int = Form(30),
    width: int = Form(640),
    height: int = Form(480),
    face_image: Optional[UploadFile] = File(None),
):
    job_id = str(uuid.uuid4())
    face_path = None

    if face_image and texture_mode == "faceswap":
        face_path = str(RENDER_OUTPUT_DIR / f"{job_id}_face.jpg")
        with open(face_path, "wb") as f:
            f.write(await face_image.read())

    JOBS[job_id] = {
        "status": "queued", "progress": 0,
        "log": [], "cosmos_response": None, "error": None,
        "vlm_analysis": None
    }

    npz_path = str(KIMODO_OUTPUT_DIR / motion_file) if motion_file else None

    background_tasks.add_task(
        run_render_job, job_id, prompt,
        cosmos_prompt or prompt,
        texture_mode, npz_path, face_path, fps, width, height
    )
    return {"job_id": job_id}


# ─── Analyze endpoint ─────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    video: Optional[UploadFile] = File(None),
    video_path: str = Form(""),           # path on server (e.g. from a render job)
    job_id: str = Form(""),               # analyze output of an existing job
    prompt: str = Form(
        "Describe this surveillance footage in detail. "
        "Identify any suspicious activity, people, vehicles, or events."
    ),
    backend: str = Form(""),              # override VLM_BACKEND for this call
):
    """
    Analyze a video with a VLM. Accepts:
      - uploaded file (video=...)
      - server-side path (video_path=...)
      - existing job output (job_id=...)
    """
    from render.vlm_analyze import analyze_video

    # Resolve video path
    resolved_path = None

    if job_id and job_id in JOBS:
        p = RENDER_OUTPUT_DIR / f"{job_id}.mp4"
        if p.exists():
            resolved_path = str(p)
        else:
            return JSONResponse({"error": f"Job {job_id} video not ready"}, status_code=400)

    elif video_path:
        p = Path(video_path)
        if not p.exists():
            return JSONResponse({"error": f"File not found: {video_path}"}, status_code=400)
        resolved_path = str(p)

    elif video:
        tmp = RENDER_OUTPUT_DIR / f"upload_{uuid.uuid4()}.mp4"
        with open(tmp, "wb") as f:
            f.write(await video.read())
        resolved_path = str(tmp)

    else:
        return JSONResponse({"error": "Provide video file, video_path, or job_id"}, status_code=400)

    chosen_backend = backend or VLM_BACKEND
    result = await analyze_video(resolved_path, prompt=prompt, backend=chosen_backend)
    result["video_path"] = resolved_path
    return result


# ─── Generate + Analyze in one call ──────────────────────────────────────────
@app.post("/generate_and_analyze")
async def generate_and_analyze(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    vlm_prompt: str = Form(
        "Describe this surveillance footage in detail. "
        "Identify any suspicious activity, people, vehicles, or events."
    ),
    texture_mode: str = Form("cosmos"),
    cosmos_prompt: str = Form(""),
    motion_file: str = Form(""),
    fps: int = Form(30),
    width: int = Form(640),
    height: int = Form(480),
    vlm_backend: str = Form(""),          # override VLM_BACKEND
    face_image: Optional[UploadFile] = File(None),
):
    """
    Render a video AND immediately analyze it with a VLM.
    Returns job_id — poll /jobs/{job_id} for status.
    The final job will have:
      - status: "done"
      - vlm_analysis: { backend, response, model, frames_used }
    """
    job_id = str(uuid.uuid4())
    face_path = None

    if face_image and texture_mode == "faceswap":
        face_path = str(RENDER_OUTPUT_DIR / f"{job_id}_face.jpg")
        with open(face_path, "wb") as f:
            f.write(await face_image.read())

    JOBS[job_id] = {
        "status": "queued", "progress": 0,
        "log": [], "cosmos_response": None, "error": None,
        "vlm_analysis": None
    }

    npz_path = str(KIMODO_OUTPUT_DIR / motion_file) if motion_file else None
    chosen_backend = vlm_backend or VLM_BACKEND

    background_tasks.add_task(
        run_render_and_analyze_job,
        job_id, prompt, cosmos_prompt or prompt,
        texture_mode, npz_path, face_path,
        fps, width, height,
        vlm_prompt, chosen_backend
    )
    return {"job_id": job_id}


# ─── Background tasks ─────────────────────────────────────────────────────────
async def run_render_job(job_id, prompt, texture_prompt, texture_mode,
                          npz_path, face_path, fps, W, H):
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        out_video = await _do_render(job, job_id, prompt, texture_prompt,
                                     texture_mode, npz_path, face_path, fps, W, H)
        job["progress"] = 100
        job["status"]   = "done"
        job["log"].append(f"Done → {out_video}")
    except Exception as e:
        job["status"] = "error"
        job["error"]  = str(e)
        job["log"].append(f"ERROR: {e}")
        import traceback; traceback.print_exc()


async def run_render_and_analyze_job(job_id, prompt, texture_prompt, texture_mode,
                                      npz_path, face_path, fps, W, H,
                                      vlm_prompt, vlm_backend):
    from render.vlm_analyze import analyze_video
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        out_video = await _do_render(job, job_id, prompt, texture_prompt,
                                     texture_mode, npz_path, face_path, fps, W, H)
        job["progress"] = 90
        job["log"].append(f"Analyzing with VLM backend: {vlm_backend}...")

        vlm_result = await analyze_video(out_video, prompt=vlm_prompt, backend=vlm_backend)
        job["vlm_analysis"] = vlm_result
        job["log"].append(f"VLM done: {vlm_result.get('model')} — {str(vlm_result.get('response',''))[:120]}...")

        job["progress"] = 100
        job["status"]   = "done"
        job["log"].append(f"Done → {out_video}")

    except Exception as e:
        job["status"] = "error"
        job["error"]  = str(e)
        job["log"].append(f"ERROR: {e}")
        import traceback; traceback.print_exc()


async def _do_render(job, job_id, prompt, texture_prompt, texture_mode,
                     npz_path, face_path, fps, W, H) -> str:
    """Shared render logic used by both /generate and /generate_and_analyze."""
    job["log"].append("Starting render pipeline...")

    # 1. Motion
    if not npz_path or not Path(npz_path).exists():
        job["log"].append(f"Generating motion: {prompt}")
        job["progress"] = 5
        npz_path = await generate_kimodo_motion(job_id, prompt)
        job["log"].append(f"Motion ready: {npz_path}")
    else:
        job["log"].append(f"Using existing motion: {Path(npz_path).name}")

    job["progress"] = 20

    # 2. Texture colors
    colors = None
    if texture_mode == "cosmos":
        job["log"].append("Querying Cosmos-Reason2 for texture colors...")
        colors, cosmos_resp = await get_cosmos_texture(texture_prompt)
        job["cosmos_response"] = cosmos_resp
        job["log"].append("Texture colors received")
    elif texture_mode == "skeleton":
        job["log"].append("Skeleton mode — CT2.5 will add texture")

    job["progress"] = 35

    # 3. Render
    out_video = str(RENDER_OUTPUT_DIR / f"{job_id}.mp4")
    job["log"].append(f"Rendering SOMA mesh ({texture_mode})...")
    await asyncio.get_event_loop().run_in_executor(
        None, render_soma_video,
        npz_path, out_video, texture_mode, colors, face_path, fps, W, H
    )
    job["log"].append("Render complete")
    return out_video


async def generate_kimodo_motion(job_id, prompt):
    out_npz = str(RENDER_OUTPUT_DIR / f"{job_id}_motion.npz")
    env = {**__import__("os").environ, "PYTHONPATH": "/kimodo:" + __import__("os").environ.get("PYTHONPATH", "")}
    proc = await asyncio.create_subprocess_exec(
        "python", "-m", "kimodo.scripts.generate",
        "--output", out_npz,
        "--bvh",
        "--no-postprocess",
        "--seed", "42",
        prompt,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=env
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Kimodo failed: {stderr.decode()[-500:]}")
    return out_npz


async def get_cosmos_texture(prompt):
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
    from render.soma_render import render
    render(npz_path=npz_path, out_video=out_video, texture_mode=texture_mode,
           colors=colors, face_path=face_path, fps=fps, W=W, H=H)


# ─── Job polling & video serving ─────────────────────────────────────────────

@app.get("/jobs")
def list_jobs():
    return {"jobs": [{"job_id": jid, **{k: v for k, v in j.items() if k != "log"}} for jid, j in JOBS.items()]}

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
