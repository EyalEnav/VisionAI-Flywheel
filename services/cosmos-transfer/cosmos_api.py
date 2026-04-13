"""
Cosmos Transfer API — lightweight FastAPI wrapper
"""
import os, uuid, asyncio, subprocess, json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Cosmos Transfer API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS: dict[str, dict] = {}

class TransferRequest(BaseModel):
    input_path: str
    output_path: str
    prompt: str
    control_weight: float = 1.0

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/transfer")
async def transfer(req: TransferRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "log": [], "output_path": req.output_path, "error": None}
    asyncio.create_task(run_transfer(job_id, req))
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JOBS[job_id]

async def run_transfer(job_id: str, req: TransferRequest):
    job = JOBS[job_id]
    job["status"] = "running"
    spec_path = f"/tmp/cosmos_spec_{job_id}.json"
    # Use a unique output dir per job to avoid collisions
    work_dir = f"/tmp/cosmos_work_{job_id}"
    os.makedirs(work_dir, exist_ok=True)

    try:
        input_name = Path(req.input_path).stem
        spec = {
            "name": input_name,
            "prompt": req.prompt,
            "video_path": req.input_path,
            "edge": {"control_weight": req.control_weight}
        }
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        job["log"].append(f"Running inference on {req.input_path}")

        proc = await asyncio.create_subprocess_exec(
            "python3", "/workspace/examples/inference.py",
            "-i", spec_path,
            "-o", work_dir,
            "--disable-guardrails",
            "--offload-guardrail-models",
            "control:edge",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        stdout, _ = await proc.communicate()
        log_text = stdout.decode()[-3000:]
        job["log"].append(log_text)

        if proc.returncode != 0:
            raise RuntimeError(f"inference.py failed (rc={proc.returncode})")

        # Find the generated video (Cosmos writes <input_name>.mp4 or similar)
        candidates = list(Path(work_dir).glob("*.mp4"))
        # exclude edge control video
        candidates = [c for c in candidates if "control_edge" not in c.name]
        if not candidates:
            raise RuntimeError(f"No output mp4 found in {work_dir}")

        # Move to desired output path
        import shutil
        shutil.move(str(candidates[0]), req.output_path)

        job["status"] = "done"
        job["output_path"] = req.output_path
        job["log"].append(f"Done → {req.output_path}")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["log"].append(f"ERROR: {e}")
    finally:
        for p in [spec_path]:
            try: os.remove(p)
            except: pass
        import shutil
        try: shutil.rmtree(work_dir)
        except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
