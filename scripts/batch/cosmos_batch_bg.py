#!/usr/bin/env python3
"""
Batch Cosmos Transfer — adds realistic backgrounds to all batch clips.
Distributes 6 background types round-robin. Skips already-done clips.
Output: /opt/dlami/nvme/cosmos_bg/<name>_bg_<scene>.mp4
"""
import json, os, time, requests

COSMOS_API   = "http://localhost:8080"
CLIP_DIR     = "/clip_storage"        # inside container
OUT_DIR      = "/cosmos_bg"           # inside container
HOST_CLIP    = "/home/ubuntu/video-search-and-summarization/deployments/data-dir/data_log/vst/clip_storage"
STATE_FILE   = "/opt/dlami/nvme/cosmos_bg/batch_state.json"

BACKGROUNDS = [
    ("city_street",
     "surveillance CCTV camera footage, outdoor city street, daytime, urban environment, "
     "concrete sidewalk, road with cars, city buildings in background, natural sunlight, "
     "realistic outdoor scene, security camera perspective from above"),
    ("park",
     "surveillance CCTV camera footage, outdoor city park, daytime, green grass, "
     "paved walking path, trees, park benches, open sky, natural sunlight, "
     "realistic outdoor urban park, security camera perspective from above"),
    ("beach",
     "surveillance CCTV camera footage, outdoor beach boardwalk, daytime, sandy beach, "
     "ocean in background, coastal promenade, open sky, bright sunlight, "
     "realistic outdoor coastal scene, security camera perspective from above"),
    ("parking_lot",
     "surveillance CCTV camera footage, outdoor parking lot, daytime, asphalt surface, "
     "parked cars, painted parking lines, open sky, urban outdoor environment, "
     "realistic outdoor scene, security camera perspective from above"),
    ("shopping_mall",
     "surveillance CCTV camera footage, outdoor shopping mall entrance, daytime, "
     "paved plaza, storefronts, shoppers, open sky, urban commercial outdoor area, "
     "realistic outdoor scene, security camera perspective from above"),
    ("night_street",
     "surveillance CCTV camera footage, outdoor city street at night, street lights, "
     "wet asphalt, urban buildings, cars with headlights, dark sky, artificial lighting, "
     "realistic outdoor night scene, security camera perspective from above"),
]

def load_state():
    try:    return json.load(open(STATE_FILE))
    except: return {}

def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2)

def submit_job(input_path, output_path, prompt):
    r = requests.post(f"{COSMOS_API}/transfer", json={
        "input_path":  input_path,
        "output_path": output_path,
        "prompt":      prompt,
        "edge_weight": 0.85,
        "vis_weight":  0.45,
        "multicontrol": True
    }, timeout=10)
    r.raise_for_status()
    return r.json()["job_id"]

def poll_job(job_id, timeout=1200):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{COSMOS_API}/jobs/{job_id}", timeout=10).json()
        status = r.get("status", "")
        if status == "done":  return True, r.get("output_path", "")
        if status == "error": return False, r.get("error", "unknown")
        time.sleep(15)
    return False, "timeout"

def main():
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    # Get all canonical clips (no _1, _test, _cosmos suffixes)
    clips = sorted([
        f[:-4] for f in os.listdir(HOST_CLIP)
        if f.endswith(".mp4")
        and not f.endswith("_1.mp4")
        and "_test" not in f
        and "_cosmos" not in f
        and not f.startswith(".")
    ])
    print(f"Clips to process: {len(clips)}", flush=True)

    state = load_state()
    done = sum(1 for s in state.values() if s.get("status") == "done")
    print(f"Already done: {done}", flush=True)

    for i, name in enumerate(clips):
        bg_id, bg_prompt = BACKGROUNDS[i % len(BACKGROUNDS)]
        output_path = f"{OUT_DIR}/{name}_bg_{bg_id}.mp4"
        host_output = f"/opt/dlami/nvme/cosmos_bg/{name}_bg_{bg_id}.mp4"

        # Skip if already done
        if state.get(name, {}).get("status") == "done":
            print(f"  [{i+1}/{len(clips)}] skip {name}", flush=True)
            continue
        # Skip if output file already exists
        if os.path.exists(host_output) and os.path.getsize(host_output) > 10000:
            print(f"  [{i+1}/{len(clips)}] file exists, marking done: {name}", flush=True)
            state[name] = {"status": "done", "output": host_output, "bg": bg_id}
            save_state(state)
            continue

        input_path = f"{CLIP_DIR}/{name}.mp4"
        print(f"\n[{i+1}/{len(clips)}] {name} → {bg_id}", flush=True)

        try:
            job_id = submit_job(input_path, output_path, bg_prompt)
            print(f"  job={job_id[:8]}...", flush=True)
            ok, result = poll_job(job_id)
            if ok:
                print(f"  ✓ DONE → {result}", flush=True)
                state[name] = {"status": "done", "output": result, "bg": bg_id}
            else:
                print(f"  ✗ ERROR: {result}", flush=True)
                state[name] = {"status": "error", "error": result}
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}", flush=True)
            state[name] = {"status": "error", "error": str(e)}

        save_state(state)
        done_now = sum(1 for s in state.values() if s.get("status") == "done")
        errors   = sum(1 for s in state.values() if s.get("status") == "error")
        print(f"  Progress: {done_now} done, {errors} errors / {len(clips)} total", flush=True)

    done_t = sum(1 for s in state.values() if s.get("status") == "done")
    err_t  = sum(1 for s in state.values() if s.get("status") == "error")
    print(f"\n=== BATCH COMPLETE: {done_t} success, {err_t} errors ===")

if __name__ == "__main__":
    main()
