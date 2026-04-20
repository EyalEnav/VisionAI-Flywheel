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
COSMOS_TRANSFER_URL = os.environ.get("COSMOS_TRANSFER_URL", "http://172.18.0.1:8080")
INSIGHTFACE_MODEL  = os.environ.get("INSIGHTFACE_MODEL", "/models/inswapper_128.onnx")

# ─── Multi-GPU Kimodo round-robin ─────────────────────────────────────────────
import itertools as _itertools
_KIMODO_GPU0 = os.environ.get("KIMODO_GPU0_URL", "")
_KIMODO_GPU1 = os.environ.get("KIMODO_GPU1_URL", "")
_KIMODO_DEFAULT = os.environ.get("KIMODO_API_URL", "http://kimodo-api:9551")

def _build_kimodo_pool():
    urls = [u for u in [_KIMODO_GPU0, _KIMODO_GPU1] if u]
    if not urls:
        urls = [_KIMODO_DEFAULT]
    return urls

_kimodo_pool = _build_kimodo_pool()
_kimodo_rr   = _itertools.cycle(_kimodo_pool)

def get_kimodo_url() -> str:
    """Return next Kimodo backend URL (round-robin across available GPUs)."""
    return next(_kimodo_rr)

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
        "vlm_analysis": None,
        "cosmos_enabled": texture_mode == "cosmos"
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

    if job_id:
        # Support "<id>_cosmos" suffix for Cosmos output
        real_id = job_id.replace("_cosmos", "")
        suffix = "_cosmos" if job_id.endswith("_cosmos") else ""
        if real_id in JOBS:
            p = RENDER_OUTPUT_DIR / f"{real_id}{suffix}.mp4"
            if p.exists():
                resolved_path = str(p)
            else:
                # fallback to non-cosmos version
                p2 = RENDER_OUTPUT_DIR / f"{real_id}.mp4"
                if p2.exists():
                    resolved_path = str(p2)
                else:
                    return JSONResponse({"error": f"Job {job_id} video not ready"}, status_code=400)
        else:
            return JSONResponse({"error": f"Job {job_id} not found"}, status_code=404)

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
        "vlm_analysis": None,
        "cosmos_enabled": texture_mode == "cosmos"
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
    if texture_mode == "clothed": texture_mode = "cosmos"  # alias
    if texture_mode in ("cosmos", "soma"):
        job["log"].append("Querying Cosmos-Reason2 for texture colors...")
        try:
            colors, cosmos_resp = await get_cosmos_texture(texture_prompt)
            job["cosmos_response"] = cosmos_resp
            job["log"].append("Texture colors received")
        except Exception as tex_e:
            job["log"].append(f"Texture fallback: {tex_e}")
            colors = parse_cosmos_colors("")
    elif texture_mode == "skeleton":
        job["log"].append("Skeleton mode — CT2.5 will add texture")

    job["progress"] = 35
    if colors is None and texture_mode not in ("skeleton", "faceswap"):
        colors = parse_cosmos_colors("")

    # 3. Render
    out_video = str(RENDER_OUTPUT_DIR / f"{job_id}.mp4")
    job["log"].append(f"Rendering SOMA mesh ({texture_mode})...")
    await asyncio.get_event_loop().run_in_executor(
        None, render_soma_video,
        npz_path, out_video, texture_mode, colors, face_path, fps, W, H
    )
    job["log"].append("Render complete")
    job["progress"] = 60

    # 4. Cosmos Transfer (Sim2Real) — if texture_mode == "cosmos"
    cosmos_out = str(RENDER_OUTPUT_DIR / f"{job_id}_cosmos.mp4")
    if texture_mode == "cosmos":
        job["log"].append("Applying Cosmos Transfer2.5 (Sim2Real)...")
        try:
            cosmos_out = await apply_cosmos_transfer(job_id, out_video, cosmos_out, texture_prompt or prompt)
            job["log"].append(f"Cosmos Transfer complete → {cosmos_out}")
        except Exception as ct_e:
            job["log"].append(f"Cosmos Transfer failed: {ct_e} — using rendered video")
            import shutil
            shutil.copy2(out_video, cosmos_out)
    else:
        import shutil
        shutil.copy2(out_video, cosmos_out)

    job["progress"] = 95
    return out_video


async def generate_kimodo_motion(job_id, prompt):
    import httpx, shutil
    out_npz = str(RENDER_OUTPUT_DIR / f"{job_id}_motion.npz")
    kimodo_url = get_kimodo_url()  # round-robin across GPU0/GPU1
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{kimodo_url}/generate",
            json={"prompt": prompt, "num_frames": 196},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Kimodo API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
    # kimodo-api saves the npz to the shared /kimodo_output volume
    src_npz = data.get("output_path") or str(KIMODO_OUTPUT_DIR / data.get("filename", ""))
    if not src_npz or not Path(src_npz).exists():
        # fallback: find latest npz in kimodo_output
        npzs = sorted(KIMODO_OUTPUT_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if not npzs:
            raise RuntimeError("Kimodo API returned no NPZ file")
        src_npz = str(npzs[-1])
    shutil.copy2(src_npz, out_npz)
    return out_npz


async def generate_person_mask(soma_video: str, mask_path: str):
    """Generate binary person mask from SOMA render (black background chroma key)."""
    import subprocess, sys
    script = Path(__file__).parent / "render" / "generate_mask.py"
    proc = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            [sys.executable, str(script), soma_video, mask_path],
            capture_output=True, text=True
        )
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Mask generation failed: {proc.stderr[:500]}")
    return mask_path


async def apply_cosmos_transfer(job_id: str, input_video: str, output_video: str, prompt: str) -> str:
    """Call the cosmos-transfer container to run Sim2Real on the rendered video."""
    import httpx, asyncio

    # Generate person mask from SOMA render
    mask_path = str(RENDER_OUTPUT_DIR / f"{job_id}_mask.mp4")
    try:
        await generate_person_mask(input_video, mask_path)
    except Exception as me:
        mask_path = None  # fallback: no mask

    payload = {
        "input_path": input_video,
        "output_path": output_video,
        "prompt": prompt,
        "control_weight": float(os.environ.get("COSMOS_EDGE_WEIGHT", "0.85")),
    }
    if mask_path and Path(mask_path).exists():
        payload["guided_mask_path"] = mask_path

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{COSMOS_TRANSFER_URL}/transfer", json=payload)
        resp.raise_for_status()
        ct_job_id = resp.json()["job_id"]

    # Poll until done
    async with httpx.AsyncClient(timeout=10) as client:
        for _ in range(900):  # up to ~30 min
            await asyncio.sleep(2)
            r = await client.get(f"{COSMOS_TRANSFER_URL}/jobs/{ct_job_id}")
            data = r.json()
            if data["status"] == "done":
                return output_video
            if data["status"] == "error":
                raise RuntimeError(data.get("error", "cosmos transfer error"))
    raise RuntimeError("Cosmos Transfer timed out after 30 minutes")


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



# ─── Cosmos Transfer — standalone endpoint ────────────────────────────────────
@app.post("/cosmos")
async def cosmos_endpoint(
    video:         Optional[UploadFile] = File(None),
    job_id:        str  = Form(""),         # existing render job id
    prompt:        str  = Form("A person walking in a photorealistic urban environment. Surveillance camera footage."),
    edge_weight:   float = Form(0.85),
    vis_weight:    float = Form(0.45),
):
    """
    Run Cosmos Transfer2.5 Sim2Real on any video.
    Accepts:
      - uploaded file  (video=...)
      - existing job   (job_id=...)
    Returns job_id — poll /jobs/{job_id} for status, GET /render/video/{job_id}_cosmos for result.
    """
    import asyncio, shutil

    # Resolve source video
    source_path = None
    cosmos_job_id = str(uuid.uuid4())

    if job_id:
        real_id = job_id.replace("_cosmos", "")
        p = RENDER_OUTPUT_DIR / f"{real_id}.mp4"
        if p.exists():
            source_path = str(p)
        else:
            return JSONResponse({"error": f"Job {job_id} video not found"}, status_code=404)
    elif video:
        tmp = RENDER_OUTPUT_DIR / f"upload_{cosmos_job_id}.mp4"
        with open(tmp, "wb") as f:
            f.write(await video.read())
        source_path = str(tmp)
    else:
        return JSONResponse({"error": "Provide video file or job_id"}, status_code=400)

    # Register job
    JOBS[cosmos_job_id] = {
        "status": "running", "progress": 0,
        "log": ["Starting Cosmos Transfer…"],
        "cosmos_response": None, "error": None,
    }

    async def _run():
        job = JOBS[cosmos_job_id]
        try:
            output_path = str(RENDER_OUTPUT_DIR / f"{cosmos_job_id}_cosmos.mp4")

            # Override env weights for this call
            old_edge = os.environ.get("COSMOS_EDGE_WEIGHT")
            old_vis  = os.environ.get("COSMOS_VIS_WEIGHT")
            os.environ["COSMOS_EDGE_WEIGHT"] = str(edge_weight)
            os.environ["COSMOS_VIS_WEIGHT"]  = str(vis_weight)

            job["log"].append(f"edge={edge_weight}  vis={vis_weight}")
            job["log"].append(f"prompt: {prompt[:80]}")
            job["progress"] = 10

            result = await apply_cosmos_transfer(cosmos_job_id, source_path, output_path, prompt)

            if old_edge is not None: os.environ["COSMOS_EDGE_WEIGHT"] = old_edge
            if old_vis  is not None: os.environ["COSMOS_VIS_WEIGHT"]  = old_vis

            job["log"].append("✅ Cosmos Transfer complete")
            job["progress"] = 100
            job["status"] = "done"
        except Exception as e:
            job["error"] = str(e)
            job["status"] = "error"
            job["log"].append(f"❌ {e}")

    asyncio.create_task(_run())
    return {"job_id": cosmos_job_id}

# ─── Job polling & video serving ─────────────────────────────────────────────
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JOBS[job_id]


@app.get("/render/video/{job_id:path}")
def get_video(job_id: str):
    # Support _cosmos suffix: /render/video/abc123_cosmos
    path = RENDER_OUTPUT_DIR / f"{job_id}.mp4"
    if not path.exists():
        return JSONResponse({"error": "video not ready"}, status_code=404)
    safe_name = job_id.replace("/", "_")
    return FileResponse(str(path), media_type="video/mp4",
                        filename=f"render_{safe_name[:12]}.mp4")
