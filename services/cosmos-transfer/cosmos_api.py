"""
Cosmos Transfer API v3.2 — multicontrol: edge (Canny) + vis (blur) guidance
No guided mask — clean transfer only.
"""
import os, uuid, asyncio, json, shutil, subprocess
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Cosmos Transfer API", version="3.2")
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
sigma_max: Optional[int] = None  # 0-200, None=default. Lower=more faithful to input


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
    JOBS[job_id] = {"status": "queued", "log": [], "output_path": req.output_path, "error": None}
    await JOB_QUEUE.put((job_id, req))
    return {"job_id": job_id, "queue_position": JOB_QUEUE.qsize()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JOBS[job_id]


@app.get("/jobs")
def list_jobs():
    return {jid: {"status": j["status"], "output": j.get("output_path", "")} for jid, j in JOBS.items()}


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


async def run_transfer(job_id: str, req: TransferRequest):
    job = JOBS[job_id]
    job["status"] = "running"
    work_dir  = f"/tmp/cosmos_work_{job_id}"
    spec_path = f"/tmp/cosmos_spec_{job_id}.json"
    os.makedirs(work_dir, exist_ok=True)

    try:
        input_name = Path(req.input_path).stem

        # Build spec — no mask, just edge+vis
        use_multi = req.multicontrol and req.vis_weight > 0.0
        spec = {
            "name": input_name,
            "prompt": req.prompt,
            "video_path": req.input_path,
            "guidance": req.guidance,
            "edge": {"control_weight": req.edge_weight},
        }
        if use_multi:
            spec["vis"] = {"control_weight": req.vis_weight}
            job["log"].append(f"[spec] multicontrol edge={req.edge_weight} vis={req.vis_weight}")
        else:
            job["log"].append(f"[spec] edge-only edge={req.edge_weight}")

if req.sigma_max is not None:
            spec["sigma_max"] = str(req.sigma_max)
            job["log"].append(f"[sigma_max] {req.sigma_max}")

        if mask_path and os.path.exists(mask_path):
            spec["guided_generation_mask"] = mask_path
            spec["guided_generation_step_threshold"] = req.guided_steps
            job["log"].append(f"[guided] mask={mask_path}, steps={req.guided_steps}")


        job["log"].append(f"[prompt] {req.prompt[:80]}")

        if req.sigma_max is not None:
            spec["sigma_max"] = req.sigma_max

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
        stdout, _ = await proc.communicate()
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
