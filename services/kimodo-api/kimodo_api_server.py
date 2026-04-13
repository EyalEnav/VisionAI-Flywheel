"""
Kimodo Generation API — nvcr.io/nvidia/pytorch:25.03-py3 (sm_120 Blackwell support)
POST /generate  { "prompt": "...", "output_filename": "optional.npz" }
GET  /health
"""
import os, sys, uuid, traceback, asyncio
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, "/workspace")
os.environ.setdefault("HF_HOME", "/hf_cache")

app = FastAPI(title="Kimodo API")

OUTPUT_DIR = Path(os.environ.get("KIMODO_OUTPUT_DIR", "/kimodo_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── lazy model cache ──────────────────────────────────────────────────────────
_model = None
_device = None

def _load_model():
    global _model, _device
    if _model is not None:
        return _model, _device
    import torch
    from kimodo.scripts.generate import load_model
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[kimodo-api] Loading model on {_device} ...")
    _model, _ = load_model(None, device=_device, default_family="Kimodo", return_resolved_name=True)
    print(f"[kimodo-api] Model loaded ✓")
    return _model, _device

# ── request schema ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    num_frames: int = 196
    output_filename: str | None = None

# ── helpers ───────────────────────────────────────────────────────────────────
def _run_sync(prompt: str, num_frames: int, out_path: str):
    import torch
    from kimodo.scripts.generate import resolve_cfg_kwargs
    import argparse

    model, device = _load_model()

    output = model(
        [prompt],
        [num_frames],
        constraint_lst=[],
        num_denoising_steps=50,
        num_samples=1,
        multi_prompt=True,
        num_transition_frames=0,
        post_processing=True,
        return_numpy=True,
    )

    # Save NPZ (same fields as CLI)
    np.savez(
        out_path,
        **{k: v for k, v in output.items() if isinstance(v, np.ndarray)}
    )
    print(f"[kimodo-api] Saved → {out_path}")

# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    import torch
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "sm": torch.cuda.get_arch_list()[-1] if torch.cuda.is_available() else "n/a",
    }

@app.post("/generate")
async def generate(req: GenerateRequest):
    filename = req.output_filename or f"{uuid.uuid4()}_motion.npz"
    out_path  = str(OUTPUT_DIR / Path(filename).name)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _run_sync, req.prompt, req.num_frames, out_path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    if not Path(out_path).exists():
        raise HTTPException(status_code=500, detail="Output file not created")
    return {"ok": True, "filename": Path(out_path).name, "output_path": out_path}
