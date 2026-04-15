"""
vlm_analyze.py — Video → VLM analysis with pluggable backends

Supported backends (set via VLM_BACKEND env var):
  - "vss"       NVIDIA VSS Agent (default) — http://localhost:8000
  - "qwen"      Qwen2.5-VL-2B-Instruct   — local HuggingFace (GPU0)
  - "qwen7b"    Qwen2.5-VL-7B-Instruct   — local HuggingFace (GPU0)
  - "openai"    GPT-4o / GPT-4-vision     — OpenAI API
  - "nim"       Any NVIDIA NIM endpoint   — custom URL

Usage:
  from render.vlm_analyze import analyze_video
  result = await analyze_video("/path/to/video.mp4", prompt="Describe this surveillance footage")
"""

import os
import asyncio
import base64
import tempfile
import httpx
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
VLM_BACKEND      = os.environ.get("VLM_BACKEND", "vss")        # vss | qwen | qwen7b | openai | nim
VSS_URL          = os.environ.get("VSS_URL", "http://localhost:8000")
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
NIM_URL          = os.environ.get("NIM_URL", "")
NIM_MODEL        = os.environ.get("NIM_MODEL", "")
HF_HOME          = os.environ.get("HF_HOME", "/hf_cache")

# Lazy-loaded Qwen model (loaded once, reused across calls)
_qwen_model      = None
_qwen_processor  = None
_qwen_model_id   = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_video(
    video_path: str,
    prompt: str = "Describe this surveillance footage in detail. Identify any suspicious activity, people, vehicles, or events.",
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
        elif backend in ("qwen", "qwen7b"):
            model_id = "Qwen/Qwen2.5-VL-7B-Instruct" if backend == "qwen7b" else "Qwen/Qwen2.5-VL-2B-Instruct"
            result = await asyncio.get_event_loop().run_in_executor(
                None, _analyze_qwen, video_path, prompt, model_id, max_frames
            )
        elif backend == "openai":
            result = await _analyze_openai(video_path, prompt, max_frames)
        elif backend == "nim":
            result = await _analyze_nim(video_path, prompt, max_frames)
        else:
            raise ValueError(f"Unknown VLM backend: {backend}")

        result["backend"] = backend
        return result

    except Exception as e:
        return {
            "backend": backend,
            "response": None,
            "model": None,
            "frames_used": 0,
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Backend: NVIDIA VSS Agent
# ─────────────────────────────────────────────────────────────────────────────

async def _analyze_vss(video_path: str, prompt: str) -> dict:
    """
    Upload video to VSS, then query the agent.
    Requires VSS running at VSS_URL (default :8000).
    """
    import shutil

    # Step 1: copy video to VSS clip storage
    vss_storage = Path("/home/ubuntu/video-search-and-summarization/deployments/data-dir"
                       "/data_log/vst/clip_storage")
    vss_storage.mkdir(parents=True, exist_ok=True)

    fname = Path(video_path).name
    dest  = vss_storage / fname
    shutil.copy2(video_path, dest)

    # Step 2: trigger VSS upload via MCP (best-effort)
    # The file presence is often enough; alternatively poll /sensors
    await asyncio.sleep(2)

    # Step 3: query VSS chat agent
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{VSS_URL}/api/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "use_knowledge_base": False
            }
        )
        data = resp.json()

    answer = data.get("choices", [{}])[0].get("message", {}).get("content", str(data))
    return {"response": answer, "model": "nvidia/vss-agent", "frames_used": -1, "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Qwen2.5-VL (local HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_qwen(video_path: str, prompt: str, model_id: str, max_frames: int) -> dict:
    """
    Run Qwen2.5-VL locally. Loads model lazily (stays in VRAM between calls).
    Requires: pip install qwen-vl-utils transformers torch
    """
    global _qwen_model, _qwen_processor, _qwen_model_id

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Load model if needed (or if backend changed)
    if _qwen_model is None or _qwen_model_id != model_id:
        print(f"[vlm_analyze] Loading {model_id}...")
        _qwen_processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_HOME)
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=HF_HOME,
        )
        _qwen_model.eval()
        _qwen_model_id = model_id
        print(f"[vlm_analyze] {model_id} loaded ✓")

    # Extract frames as base64
    frames_b64 = _extract_frames_b64(video_path, max_frames)
    frames_used = len(frames_b64)

    # Build message with video frames as images
    content = []
    for f_b64 in frames_b64:
        content.append({
            "type": "image",
            "image": f"data:image/jpeg;base64,{f_b64}"
        })
    content.append({"type": "text", "text": prompt})

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
        generated_ids = _qwen_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = _qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return {
        "response": response,
        "model": model_id,
        "frames_used": frames_used,
        "error": None
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backend: OpenAI GPT-4o
# ─────────────────────────────────────────────────────────────────────────────

async def _analyze_openai(video_path: str, prompt: str, max_frames: int) -> dict:
    frames_b64 = _extract_frames_b64(video_path, max_frames)
    content = []
    for f_b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f_b64}", "detail": "low"}
        })
    content.append({"type": "text", "text": prompt})

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 512
            }
        )
        data = resp.json()

    answer = data["choices"][0]["message"]["content"]
    return {"response": answer, "model": "gpt-4o", "frames_used": len(frames_b64), "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# Backend: NVIDIA NIM (generic OpenAI-compatible endpoint)
# ─────────────────────────────────────────────────────────────────────────────

async def _analyze_nim(video_path: str, prompt: str, max_frames: int) -> dict:
    """
    Any OpenAI-compatible NIM endpoint.
    Set NIM_URL and NIM_MODEL env vars.
    """
    frames_b64 = _extract_frames_b64(video_path, max_frames)
    content = []
    for f_b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f_b64}"}
        })
    content.append({"type": "text", "text": prompt})

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{NIM_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ.get('NGC_CLI_API_KEY', '')}"},
            json={
                "model": NIM_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 512
            }
        )
        data = resp.json()

    answer = data["choices"][0]["message"]["content"]
    return {"response": answer, "model": NIM_MODEL, "frames_used": len(frames_b64), "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frames_b64(video_path: str, max_frames: int = 16) -> list[str]:
    """
    Extract up to max_frames evenly spaced frames from a video.
    Returns list of base64-encoded JPEG strings.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    indices = _even_indices(total, max_frames)
    frames_b64 = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Resize to 640×360 max for efficiency
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames_b64.append(base64.b64encode(buf.tobytes()).decode())

    cap.release()
    return frames_b64


def _even_indices(total: int, n: int) -> list[int]:
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(step * i) for i in range(n)]
