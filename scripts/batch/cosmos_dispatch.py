#!/usr/bin/env python3
"""Cosmos Transfer Dispatcher v2 — 4x GPU0 + 3x GPU1 = 7 parallel workers"""
import json, sys, requests, time, argparse
from pathlib import Path

WORKERS = [
    {"port": 8080, "gpu": 1, "name": "gpu1-main"},
    {"port": 8081, "gpu": 0, "name": "gpu0-a"},
    {"port": 8082, "gpu": 0, "name": "gpu0-b"},
    {"port": 8083, "gpu": 0, "name": "gpu0-c"},
    {"port": 8084, "gpu": 1, "name": "gpu1-b"},
    {"port": 8085, "gpu": 1, "name": "gpu1-c"},
    {"port": 8087, "gpu": 0, "name": "gpu0-d"},
]

def get_status(port):
    try:
        r = requests.get(f"http://localhost:{port}/jobs", timeout=3)
        jobs = r.json()
        return sum(1 for j in jobs.values() if j["status"] in ("running", "queued"))
    except:
        return 999

def pick_worker():
    loads = [(get_status(w["port"]), w["gpu"], w["port"], w) for w in WORKERS]
    loads.sort()
    return loads[0][3]

def submit(worker, job):
    r = requests.post(f"http://localhost:{worker['port']}/transfer", json=job, timeout=10)
    return r.json()

def status_all():
    for w in WORKERS:
        try:
            r = requests.get(f"http://localhost:{w['port']}/jobs", timeout=3)
            jobs = r.json()
            running = sum(1 for j in jobs.values() if j["status"] == "running")
            queued  = sum(1 for j in jobs.values() if j["status"] == "queued")
            done    = sum(1 for j in jobs.values() if j["status"] == "done")
            errors  = sum(1 for j in jobs.values() if j["status"] == "error")
            print(f"{w['name']:12} GPU{w['gpu']} port:{w['port']}  R:{running} Q:{queued} D:{done} E:{errors}")
        except:
            print(f"{w['name']:12} UNREACHABLE")

def wait_all():
    print("Monitoring (Ctrl+C to stop)...")
    while True:
        all_done = True
        parts = []
        for w in WORKERS:
            try:
                r = requests.get(f"http://localhost:{w['port']}/jobs", timeout=3)
                jobs = r.json()
                rr = sum(1 for j in jobs.values() if j["status"] == "running")
                qq = sum(1 for j in jobs.values() if j["status"] == "queued")
                dd = sum(1 for j in jobs.values() if j["status"] == "done")
                parts.append(f"{w['name']}:R{rr}Q{qq}D{dd}")
                if rr > 0 or qq > 0:
                    all_done = False
            except:
                parts.append(f"{w['name']}:ERR")
        print("\r" + " | ".join(parts) + "   ", end="", flush=True)
        if all_done:
            print("\nAll done!")
            break
        time.sleep(15)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cosmos Transfer Dispatcher")
    parser.add_argument("jobs", nargs="*", help="JSON job files to dispatch")
    parser.add_argument("--wait", action="store_true", help="Monitor until all complete")
    parser.add_argument("--status", action="store_true", help="Show worker status")
    args = parser.parse_args()

    if args.status:
        status_all()
        sys.exit(0)

    job_list = []
    if args.jobs:
        for jf in args.jobs:
            job_list.append(json.loads(Path(jf).read_text()))
    else:
        for line in sys.stdin:
            line = line.strip()
            if line:
                job_list.append(json.loads(line))

    print(f"Dispatching {len(job_list)} jobs across {len(WORKERS)} workers...")
    for job in job_list:
        w = pick_worker()
        result = submit(w, job)
        name = job.get("output_path", "").split("/")[-1]
        print(f"  -> {w['name']:12} port:{w['port']}  {name}  job:{result.get('job_id','?')[:8]}")

    if args.wait:
        wait_all()
