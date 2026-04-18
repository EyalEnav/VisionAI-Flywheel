"""
Cosmos Transfer API v2.1 — multicontrol via JSON spec keys (auto-detected)
"""
import os, uuid, asyncio, json, shutil
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Cosmos Transfer API", version="2.1")
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

        if req.multicontrol:
            # Both keys in JSON → Cosmos auto-selects multicontrol model
            spec = {
                "name": input_name,
                "prompt": req.prompt,
                "video_path": req.input_path,
                "edge": {"control_weight": req.edge_weight},
                "vis":  {"control_weight": req.vis_weight},
            }
            # No positional arg needed — Cosmos auto-detects from spec keys
            control_args = []
        else:
            spec = {
                "name": input_name,
                "prompt": req.prompt,
                "video_path": req.input_path,
                "edge": {"control_weight": req.edge_weight},
            }
            control_args = ["control:edge"]

        with open(spec_path, "w") as f:
            json.dump(spec, f)

        mode = "multicontrol(edge+vis)" if req.multicontrol else "edge-only"
        job["log"].append(f"[{mode}] {req.input_path}")
        job["log"].append(f"Prompt: {req.prompt[:80]}")

        cmd = [
            "python3", "/workspace/examples/inference.py",
            "-i", spec_path,
            "-o", work_dir,
            "--disable-guardrails",
            "--offload-guardrail-models",
        ] + control_args

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
