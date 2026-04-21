#!/usr/bin/env python3
"""
Cosmos Transfer batch for the 12 new incident renders.
V2 - adds clothing description to all background prompts.
"""
import json, os, time, requests, subprocess

COSMOS_API = "http://localhost:8080"
RENDER_DIR_HOST = "/home/ubuntu/render_output"
RENDER_DIR = "/render_output"
OUT_DIR    = "/opt/dlami/nvme/cosmos_full"
STATE_FILE = "/opt/dlami/nvme/cosmos_full/new12_state.json"

# Clothing description injected into all prompts
CLOTHING = (
    "person wearing a red hoodie sweatshirt and dark blue jeans, white sneakers"
)

# Job UUID → clip name mapping
CLIPS = {
    "fbcca3db-7749-4fe9-b9a2-2613c756f252": "run_and_fall",
    "7c016ce6-f72a-4b30-b92a-90c8d638ef81": "limp_walk",
    "6cbe1089-b955-48b9-9223-ef88c24b045c": "crawling",
    "f14ef7f1-636f-4c0d-8055-e525c87f2144": "stumble_drunk",
    "b9d2c38b-f076-44a0-a207-a14699aacca6": "punch_fight",
    "f84c70ff-81c7-49ec-93d0-bc83749d7029": "assault_grab",
    "57df90e5-e2b5-4872-aaab-fb4f2a17991d": "person_shoved",
    "e4ada141-f0f8-46e8-8530-57ed5c6aa66e": "running_escape",
    "7484ddb6-f45e-4bef-94ec-1924ea291c27": "collapse_sudden",
    "c4796ccc-4e3f-4b71-bacf-93e9b181ef2c": "drag_body",
    "8e5e4ef3-8f13-4e84-9ae4-86f4f01455db": "person_crouching_hiding",
    "28ec8b03-bebe-4de1-89ab-7e16e2380696": "waving_help",
}

def bg_with_clothing(base_prompt):
    """Inject clothing description into background prompt."""
    return base_prompt + f", {CLOTHING}"

BACKGROUNDS_BASE = [
    ("city_street",
     "surveillance CCTV camera footage, outdoor city street, daytime, urban environment, "
     "concrete sidewalk, road with cars, city buildings in background, natural sunlight, "
     "realistic outdoor scene, security camera perspective from above"),
    ("park",
     "surveillance CCTV camera footage, outdoor city park, daytime, green grass, "
     "paved walking path, trees, park benches, open sky, natural sunlight, "
     "realistic outdoor scene, security camera perspective from above"),
    ("parking_lot",
     "surveillance CCTV camera footage, outdoor parking lot, daytime, asphalt surface, "
     "parked cars, painted parking lines, open sky, urban outdoor environment, "
     "realistic outdoor scene, security camera perspective from above"),
    ("shopping_mall",
     "surveillance CCTV camera footage, indoor shopping mall, bright overhead lighting, "
     "tiled floor, shoppers walking, storefronts, realistic indoor commercial space, "
     "security camera perspective from above"),
    ("subway",
     "surveillance CCTV camera footage, underground subway station, tiled platform, "
     "subway train doors, fluorescent lighting, commuters, realistic indoor transit scene, "
     "security camera perspective from above"),
    ("night_street",
     "surveillance CCTV camera footage, outdoor city street at night, street lights, "
     "wet asphalt, urban buildings, cars with headlights, dark sky, artificial lighting, "
     "realistic outdoor night scene, security camera perspective from above"),
]

# Inject clothing into all background prompts
BACKGROUNDS = [(bg_id, bg_with_clothing(prompt)) for bg_id, prompt in BACKGROUNDS_BASE]

def load_state():
    try: return json.load(open(STATE_FILE))
    except: return {}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    json.dump(state, open(STATE_FILE, "w"), indent=2)

def wait_for_queue_clear():
    """Wait until existing cosmos jobs finish."""
    print("Waiting for existing cosmos queue to clear...", flush=True)
    while True:
        r = requests.get(f"{COSMOS_API}/jobs", timeout=10).json()
        active = [j for j in r.values() if j.get("status") in ("queued", "running")]
        if not active:
            print("Queue clear!", flush=True)
            return
        print(f"  {len(active)} jobs still active, waiting 30s...", flush=True)
        time.sleep(30)

def submit_job(input_path, output_path, prompt):
    payload = {
        "input_path": input_path,
        "output_path": output_path,
        "prompt": prompt,
        "edge_weight": 0.85,
        "vis_weight": 0.45,
        "guided_steps": 20,
        "guidance": 5.0,
        "multicontrol": True,
    }
    r = requests.post(f"{COSMOS_API}/transfer", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["job_id"]

def poll_job(job_id, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{COSMOS_API}/jobs/{job_id}", timeout=10).json()
        status = r.get("status", "")
        if status == "done": return True, r.get("output_path", "")
        if status == "error": return False, r.get("error", "unknown")
        time.sleep(15)
    return False, "timeout"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Show clothing being used
    print(f"Clothing: {CLOTHING}", flush=True)
    
    # Wait for existing queue
    wait_for_queue_clear()
    
    state = load_state()
    total = len(CLIPS) * len(BACKGROUNDS)
    done_count = 0
    
    for job_uuid, clip_name in CLIPS.items():
        input_host = f"{RENDER_DIR}/{job_uuid}.mp4"
        input_host_check = f"{RENDER_DIR_HOST}/{job_uuid}.mp4"
        if not os.path.exists(input_host_check):
            print(f"Missing render: {clip_name} ({job_uuid})", flush=True)
            continue
        
        for bg_id, bg_prompt in BACKGROUNDS:
            key = f"{clip_name}_{bg_id}"
            output_host_check = f"{OUT_DIR}/{clip_name}_bg_{bg_id}.mp4"
            output_host = f"/cosmos_full/{clip_name}_bg_{bg_id}.mp4"
            
            if state.get(key, {}).get("status") == "done":
                print(f"  skip {key}", flush=True)
                done_count += 1
                continue
            
            if os.path.exists(output_host_check) and os.path.getsize(output_host_check) > 10000:
                print(f"  file exists: {key}", flush=True)
                state[key] = {"status": "done", "output": output_host_check}
                save_state(state)
                done_count += 1
                continue
            
            print(f"\n[{done_count+1}/{total}] {clip_name} -> {bg_id}", flush=True)
            
            try:
                job_id = submit_job(input_host, output_host, bg_prompt)
                print(f"  job={job_id[:8]}...", flush=True)
                ok, result = poll_job(job_id)
                if ok:
                    state[key] = {"status": "done", "output": output_host_check}
                    print(f"  done: {output_host}", flush=True)
                    done_count += 1
                else:
                    state[key] = {"status": "error", "error": str(result)}
                    print(f"  error: {result}", flush=True)
                save_state(state)
            except Exception as e:
                print(f"  exception: {e}", flush=True)
                state[key] = {"status": "error", "error": str(e)}
                save_state(state)
    
    done_final = sum(1 for s in state.values() if s.get("status")=="done")
    errors = sum(1 for s in state.values() if s.get("status")=="error")
    print(f"\n{'='*50}", flush=True)
    print(f"BATCH COMPLETE: {done_final}/{total} done, {errors} errors", flush=True)

if __name__ == "__main__":
    main()
