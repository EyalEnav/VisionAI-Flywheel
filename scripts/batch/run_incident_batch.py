#!/usr/bin/env python3
"""
Phase 1: Generate motion with Kimodo for all 12 incidents
Phase 2: Render with soma_render
Phase 3: Dispatch to Cosmos with all background prompts
"""
import json, requests, time, subprocess, os, sys
from pathlib import Path

BREV = "3.147.193.200"
KIMODO_URL = f"http://{BREV}:9551"
RENDER_URL = f"http://{BREV}:9001"

SSH = ["ssh", "-i", "/app/.agents/kostya_ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{BREV}"]

def ssh(cmd):
    result = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=60)
    return result.stdout.strip()

def kimodo_generate(prompt, name, num_frames=150, fps=30):
    """Generate motion NPZ via Kimodo API"""
    payload = {
        "prompt": prompt,
        "num_frames": num_frames,
        "fps": fps,
        "output_name": name
    }
    r = requests.post(f"{KIMODO_URL}/generate", json=payload, timeout=300)
    return r.json()

def check_kimodo_job(job_id):
    r = requests.get(f"{KIMODO_URL}/jobs/{job_id}", timeout=10)
    return r.json()

# Load incidents
with open("/tmp/incident_batch.json") as f:
    incidents = json.load(f)

print(f"=== Phase 1: Kimodo Motion Generation ({len(incidents)} clips) ===")

jobs = {}
for inc in incidents:
    print(f"  Submitting: {inc['name']} ...", end=" ")
    try:
        result = kimodo_generate(inc["kimodo_prompt"], inc["name"], num_frames=150)
        job_id = result.get("job_id") or result.get("id")
        jobs[inc["name"]] = {"job_id": job_id, "incident": inc}
        print(f"job_id={str(job_id)[:8]}")
    except Exception as e:
        print(f"ERROR: {e}")
        jobs[inc["name"]] = {"job_id": None, "incident": inc, "error": str(e)}
    time.sleep(0.5)

# Save job state
with open("/tmp/incident_jobs.json", "w") as f:
    json.dump(jobs, f, indent=2)

print(f"\nSubmitted {sum(1 for j in jobs.values() if j['job_id'])} / {len(incidents)} jobs")
print("Jobs saved to /tmp/incident_jobs.json")
print("\nMonitor with: python3 /tmp/check_incident_progress.py")
