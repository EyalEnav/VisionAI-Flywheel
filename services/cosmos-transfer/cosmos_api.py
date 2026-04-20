"""
Cosmos Transfer API v4.1 — multicontrol: edge (Canny) + vis (blur) guidance
Supports longer videos via num_video_frames_per_chunk and max_frames.
Added: /preview endpoint + live chunk monitoring.
"""
import os, uuid, asyncio, json, shutil, subprocess, base64
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Cosmos Transfer API", version="4.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS: dict[str, dict] = {}
JOB_QUEUE = None


class TransferRequest(BaseModel):
    input_path: str
    output_path: str
    prompt: str
    edge_weight: float = 0.85
    vis_weight: float = 0.45
    multicontrol: bool = True
    guidance: float = 5.0
    sigma_max: Optional[int] = None
    max_frames: Optional[int] = None
    num_frames_per_chunk: int = 93
    num_steps: int = 35
    resolution: str = "720"


@app.on_event("startup")
async def startup():
    global JOB_QUEUE
    JOB_QUEUE = asyncio.Queue()
    asyncio.create_task(job_worker())


@app.get("/health")
def health():
    return {"ok": True, "queued": JOB_QUEUE.qsize() if JOB_QUEUE else 0}


@app.post("/transfer")
async def transfer(req: TransferRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "queued",
        "log": [],
        "output_path": req.output_path,
        "error": None,
        "preview_frames": [],   # list of base64 JPEG thumbnails
        "chunks_done": 0,
    }
    await JOB_QUEUE.put((job_id, req))
    return {"job_id": job_id, "queue_position": JOB_QUEUE.qsize()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error": "not found"}, status_code=404)
    j = dict(JOBS[job_id])
    j.pop("preview_frames", None)  # don't include big b64 in normal status
    return j


@app.get("/jobs")
def list_jobs():
    return {jid: {"status": j["status"], "output": j.get("output_path", ""), "chunks_done": j.get("chunks_done", 0)} for jid, j in JOBS.items()}


@app.get("/preview/{job_id}")
def get_preview(job_id: str, frame_index: int = -1):
    """Return latest (or specific) preview frame as JPEG image."""
    if job_id not in JOBS:
        return JSONResponse({"error": "not found"}, status_code=404)
    frames = JOBS[job_id].get("preview_frames", [])
    if not frames:
        return JSONResponse({"error": "no preview yet", "chunks_done": JOBS[job_id].get("chunks_done", 0)}, status_code=202)
    idx = frame_index if frame_index >= 0 else len(frames) - 1
    jpeg_b64 = frames[min(idx, len(frames)-1)]
    jpeg_bytes = base64.b64decode(jpeg_b64)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/preview/{job_id}/all")
def get_all_previews(job_id: str):
    """Return list of all preview frame base64 strings."""
    if job_id not in JOBS:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "chunks_done": JOBS[job_id].get("chunks_done", 0),
        "frames": JOBS[job_id].get("preview_frames", []),
    }


async def job_worker():
    while True:
        job_id, req = await JOB_QUEUE.get()
        try:
            await run_transfer(job_id, req)
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
        finally:
            JOB_QUEUE.task_done()


def _extract_frame_b64(video_path: str, frame_num: int = 30) -> Optional[str]:
    """Extract a frame from video and return as base64 JPEG."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fn = min(frame_num, max(0, total // 2))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        # resize to max 640px width for thumbnail
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode()
    except Exception as e:
        return None


async def _chunk_watcher(job_id: str, work_dir: str, proc):
    """Watch work_dir for new chunk mp4 files and extract preview frames."""
    seen = set()
    while proc.returncode is None:
        await asyncio.sleep(10)
        try:
            for mp4 in Path(work_dir).glob("*.mp4"):
                key = str(mp4)
                if key not in seen and mp4.stat().st_size > 10000:
                    seen.add(key)
                    # small delay to make sure file is fully written
                    await asyncio.sleep(2)
                    b64 = _extract_frame_b64(str(mp4))
                    if b64:
                        JOBS[job_id]["preview_frames"].append(b64)
                        JOBS[job_id]["chunks_done"] += 1
                        JOBS[job_id]["log"].append(f"[preview] chunk {JOBS[job_id]['chunks_done']} frame captured")
        except Exception:
            pass


async def run_transfer(job_id: str, req: TransferRequest):
    job = JOBS[job_id]
    job["status"] = "running"
    work_dir  = f"/tmp/cosmos_work_{job_id}"
    spec_path = f"/tmp/cosmos_spec_{job_id}.json"
    os.makedirs(work_dir, exist_ok=True)

    try:
        input_name = Path(req.input_path).stem

        use_multi = req.multicontrol and req.vis_weight > 0.0
        spec = {
            "name": input_name,
            "prompt": req.prompt,
            "video_path": req.input_path,
            "guidance": req.guidance,
            "resolution": req.resolution,
            "num_video_frames_per_chunk": req.num_frames_per_chunk,
            "num_steps": req.num_steps,
            "edge": {"control_weight": req.edge_weight},
        }

        if req.max_frames is not None:
            spec["max_frames"] = req.max_frames

        if use_multi:
            spec["vis"] = {"control_weight": req.vis_weight}
            job["log"].append(f"[spec] multicontrol edge={req.edge_weight} vis={req.vis_weight}")
        else:
            job["log"].append(f"[spec] edge-only edge={req.edge_weight}")

        if req.sigma_max is not None:
            spec["sigma_max"] = str(req.sigma_max)
            job["log"].append(f"[sigma_max] {req.sigma_max}")

        job["log"].append(f"[frames] per_chunk={req.num_frames_per_chunk}, max_frames={req.max_frames}, resolution={req.resolution}")
        job["log"].append(f"[prompt] {req.prompt[:80]}")

        with open(spec_path, "w") as f:
            json.dump(spec, f)

        cmd = [
            "python3", "/workspace/examples/inference.py",
            "-i", spec_path,
            "-o", work_dir,
            "--disable-guardrails",
            "--offload-guardrail-models",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        # Start chunk watcher in background
        watcher_task = asyncio.create_task(_chunk_watcher(job_id, work_dir, proc))

        stdout, _ = await proc.communicate()
        watcher_task.cancel()

        log_text = stdout.decode()[-4000:]
        job["log"].append(log_text)

        if proc.returncode != 0:
            raise RuntimeError(f"inference.py rc={proc.returncode}\n{log_text[-400:]}")

        candidates = [
            c for c in Path(work_dir).glob("*.mp4")
            if "control_edge" not in c.name and "control_vis" not in c.name
        ]
        if not candidates:
            raise RuntimeError(f"No output mp4 in {work_dir}. Files: {list(Path(work_dir).iterdir())}")

        # Extract final preview from output
        final_b64 = _extract_frame_b64(str(candidates[0]))
        if final_b64:
            job["preview_frames"].append(final_b64)

        os.makedirs(os.path.dirname(req.output_path) or ".", exist_ok=True)
        shutil.move(str(candidates[0]), req.output_path)
        job["status"] = "done"
        job["output_path"] = req.output_path
        job["log"].append(f"Done -> {req.output_path}")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["log"].append(f"ERROR: {e}")
    finally:
        try: os.remove(spec_path)
        except: pass
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
