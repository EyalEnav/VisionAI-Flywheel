"""
vlm_analyze.py — Video → VLM analysis with pluggable backends

Supported backends (set via VLM_BACKEND env var or per-call override):
  "vss"     NVIDIA VSS Agent (default)          — http://localhost:8000
  "vllm"    vLLM OpenAI-compatible server       — http://localhost:8090  (NGC nvcr.io/nvidia/vllm)
  "qwen3"   Qwen3-VL-2B-Instruct in-process     — local HuggingFace GPU (recommended)
  "qwen"    Qwen2.5-VL-2B-Instruct in-process   — local HuggingFace GPU
  "qwen7b"  Qwen2.5-VL-7B-Instruct in-process   — local HuggingFace GPU
  "openai"  GPT-4o / GPT-4-vision               — OpenAI API
  "nim"     Any OpenAI-compatible NIM endpoint  — custom NIM_URL

Usage:
  from render.vlm_analyze import analyze_video, list_vllm_models
  result = await analyze_video("/path/to/video.mp4", backend="vllm")
  models = await list_vllm_models()   # → ["Qwen/Qwen2.5-VL-2B-Instruct", ...]
"""

import os
import asyncio
import base64
import httpx
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
VLM_BACKEND      = os.environ.get("VLM_BACKEND", "vss")
VSS_URL          = os.environ.get("VSS_URL", "http://localhost:8000")
COSMOS_REASON_URL = os.environ.get("COSMOS_REASON_URL", "http://172.18.0.1:30082")
COSMOS_REASON_MODEL = "nvidia/cosmos-reason2-8b"
VLLM_URL         = os.environ.get("VLLM_URL", "http://localhost:8090")
VLLM_MODEL       = os.environ.get("VLLM_MODEL", "vllm-local")   # served-model-name in vLLM
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
NIM_URL          = os.environ.get("NIM_URL", "")
NIM_MODEL        = os.environ.get("NIM_MODEL", "")
HF_HOME          = os.environ.get("HF_HOME", "/hf_cache")

AVAILABLE_BACKENDS = ["vss", "vllm", "qwen3", "qwen", "qwen7b", "openai", "nim"]

# Lazy-loaded in-process Qwen model
_qwen_model      = None
_qwen_processor  = None
_qwen_model_id   = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_video(
    video_path: str,
    prompt: str = (
        "Describe this surveillance footage in detail. "
        "Identify any suspicious activity, people, vehicles, or events."
    ),
    backend: str | None = None,
    max_frames: int = 16,
) -> dict:
    """
    Analyze a video file with a VLM.

    Returns:
        {
          "backend": str,
          "response": str,
          "model": str,
          "frames_used": int,
          "error": str | None
        }
    """
    backend = backend or VLM_BACKEND

    try:
        if backend == "vss":
            result = await _analyze_vss(video_path, prompt)
        elif backend == "vllm":
            result = await _analyze_vllm(video_path, prompt, max_frames)
        elif backend == "qwen3":
            result = await asyncio.get_event_loop().run_in_executor(
                None, _analyze_qwen3, video_path, prompt,
                "Qwen/Qwen3-VL-2B-Instruct", max_frames
            )
        elif backend in ("qwen", "qwen7b"):
            model_id = (
                "Qwen/Qwen2.5-VL-7B-Instruct"
                if backend == "qwen7b"
                else "Qwen/Qwen2.5-VL-2B-Instruct"
            )
            result = await asyncio.get_event_loop().run_in_executor(
                None, _analyze_qwen_local, video_path, prompt, model_id, max_frames
            )
        elif backend == "openai":
            result = await _analyze_openai(video_path, prompt, max_frames)
        elif backend == "nim":
            result = await _analyze_nim(video_path, prompt, max_frames)
        else:
            raise ValueError(f"Unknown VLM backend: '{backend}'. "
                             f"Available: {AVAILABLE_BACKENDS}")

        result["backend"] = backend
        return result

    except Exception as e:
        return {
            "backend": backend,
            "response": None,
            "model": None,
            "frames_used": 0,
            "error": str(e),
        }


async def list_vllm_models() -> list[str]:
    """
    Query the running vLLM server for loaded models.
    Returns [] if vLLM is not running.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{VLLM_URL}/v1/models")
            data = resp.json()
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


async def vllm_health() -> dict:
    """Return vLLM server health info."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{VLLM_URL}/health")
        models = await list_vllm_models()
        return {"ok": resp.status_code == 200, "models": models}
    except Exception as e:
        return {"ok": False, "models": [], "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Backend: vLLM (NGC nvcr.io/nvidia/vllm — OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

async def _analyze_vllm(video_path: str, prompt: str, max_frames: int) -> dict:
    """
    Send video frames to a locally running vLLM server.
    vLLM serves an OpenAI-compatible /v1/chat/completions endpoint.
    Supports any vision model loaded in vLLM (Qwen2.5-VL, LLaVA, InternVL, etc.)
    """
    frames_b64 = _extract_frames_b64(video_path, max_frames)

    # Build content: interleave frames + text prompt
    content = []
    for f_b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{f_b64}",
                "detail": "low",
            }
        })
    content.append({"type": "text", "text": prompt})

    # Get actual model name from server (or fall back to env var)
    models = await list_vllm_models()
    model_name = models[0] if models else VLLM_MODEL

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 512,
                "temperature": 0.3,
            }
        )
        data = resp.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    answer = data["choices"][0]["message"]["content"]
    usage  = data.get("usage", {})

    return {
        "response": answer,
        "model": model_name,
        "frames_used": len(frames_b64),
        "usage": usage,
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backend: NVIDIA VSS Agent
# ─────────────────────────────────────────────────────────────────────────────

async def _analyze_vss(video_path: str, prompt: str, max_frames: int = 8) -> dict:
    """
    Analyze video using nvidia/cosmos-reason2-8b VLM directly (vision model).
    Sends extracted frames as image_url content — no VST/MCP dependency.
    """
    frames_b64 = _extract_frames_b64(video_path, max_frames)

    msg_content = []
    for f_b64 in frames_b64:
        msg_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f_b64}"}
        })
    msg_content.append({"type": "text", "text": prompt})

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{COSMOS_REASON_URL}/v1/chat/completions",
            json={
                "model": COSMOS_REASON_MODEL,
                "messages": [{"role": "user", "content": msg_content}],
                "max_tokens": 512,
                "temperature": 0.3,
            }
        )
        data = resp.json()

    answer = data.get("choices", [{}])[0].get("message", {}).get("content", str(data))
    return {
        "response": answer,
        "model": COSMOS_REASON_MODEL,
        "frames_used": len(frames_b64),
        "error": None,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Backend: Qwen3-VL-2B-Instruct (in-process HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

# Separate lazy cache for Qwen3
_qwen3_model     = None
_qwen3_processor = None

def _analyze_qwen3(
    video_path: str, prompt: str, model_id: str, max_frames: int
) -> dict:
    """
    Run Qwen3-VL in-process. Uses new Qwen3VLForConditionalGeneration class.
    Requires: pip install git+https://github.com/huggingface/transformers
    (transformers >= 4.57 not yet on PyPI as of May 2025)
    """
    global _qwen3_model, _qwen3_processor

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    if _qwen3_model is None:
        print(f"[vlm_analyze] Loading {model_id}…")
        _qwen3_processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_HOME)
        _qwen3_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # recommended by Qwen team
            device_map="auto",
            cache_dir=HF_HOME,
        )
        _qwen3_model.eval()
        print(f"[vlm_analyze] {model_id} ready ✓")

    frames_b64 = _extract_frames_b64(video_path, max_frames)
    content = [
        {"type": "image", "image": f"data:image/jpeg;base64,{f}"}
        for f in frames_b64
    ] + [{"type": "text", "text": prompt}]

    messages = [{"role": "user", "content": content}]

    # Qwen3-VL uses return_dict=True, return_tensors="pt" API
    inputs = _qwen3_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_qwen3_model.device)

    with torch.inference_mode():
        gen_ids = _qwen3_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            do_sample=True,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    response = _qwen3_processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return {
        "response": response,
        "model": model_id,
        "frames_used": len(frames_b64),
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Qwen2.5-VL in-process (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_qwen_local(
    video_path: str, prompt: str, model_id: str, max_frames: int
) -> dict:
    """
    Run Qwen2.5-VL in-process. Model stays in VRAM between calls (lazy load).
    Slower first call, fast subsequent calls.
    """
    global _qwen_model, _qwen_processor, _qwen_model_id

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    if _qwen_model is None or _qwen_model_id != model_id:
        print(f"[vlm_analyze] Loading {model_id}…")
        _qwen_processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_HOME)
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=HF_HOME,
        )
        _qwen_model.eval()
        _qwen_model_id = model_id
        print(f"[vlm_analyze] {model_id} ready ✓")

    frames_b64 = _extract_frames_b64(video_path, max_frames)
    content = [
        {"type": "image", "image": f"data:image/jpeg;base64,{f}"}
        for f in frames_b64
    ] + [{"type": "text", "text": prompt}]

    messages = [{"role": "user", "content": content}]
    text = _qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_qwen_model.device)

    with torch.inference_mode():
        gen_ids = _qwen_model.generate(
            **inputs, max_new_tokens=512, temperature=0.3, do_sample=True
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, gen_ids)
    ]
    response = _qwen_processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return {
        "response": response,
        "model": model_id,
        "frames_used": len(frames_b64),
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backend: OpenAI GPT-4o
# ─────────────────────────────────────────────────────────────────────────────

async def _analyze_openai(video_path: str, prompt: str, max_frames: int) -> dict:
    frames_b64 = _extract_frames_b64(video_path, max_frames)
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{f}",
                "detail": "low",
            },
        }
        for f in frames_b64
    ] + [{"type": "text", "text": prompt}]

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 512,
            },
        )
        data = resp.json()

    answer = data["choices"][0]["message"]["content"]
    return {
        "response": answer,
        "model": "gpt-4o",
        "frames_used": len(frames_b64),
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backend: NVIDIA NIM (generic OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

async def _analyze_nim(video_path: str, prompt: str, max_frames: int) -> dict:
    frames_b64 = _extract_frames_b64(video_path, max_frames)
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames_b64
    ] + [{"type": "text", "text": prompt}]

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{NIM_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('NGC_CLI_API_KEY', '')}",
            },
            json={
                "model": NIM_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 512,
            },
        )
        data = resp.json()

    answer = data["choices"][0]["message"]["content"]
    return {
        "response": answer,
        "model": NIM_MODEL,
        "frames_used": len(frames_b64),
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frames_b64(video_path: str, max_frames: int = 16) -> list[str]:
    """Extract up to max_frames evenly spaced frames using ffmpeg → list of base64 JPEG strings."""
    import subprocess, tempfile, glob, os

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "frame_%04d.jpg")
        r = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vf", "fps=2,scale=640:-1",
             "-vframes", str(max_frames * 4), "-q:v", "3", out, "-y", "-loglevel", "error"],
            capture_output=True, timeout=30
        )
        if r.returncode != 0:
            raise ValueError(f"Could not read video: {video_path}")
        frames = sorted(glob.glob(os.path.join(tmpdir, "*.jpg")))
        if not frames:
            raise ValueError(f"Could not read video: {video_path}")
        if len(frames) > max_frames:
            step = len(frames) / max_frames
            frames = [frames[int(i * step)] for i in range(max_frames)]
        out_b64 = []
        for f in frames:
            with open(f, "rb") as fp:
                out_b64.append(base64.b64encode(fp.read()).decode())
        return out_b64


def _even_indices(total: int, n: int) -> list[int]:
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(step * i) for i in range(n)]
