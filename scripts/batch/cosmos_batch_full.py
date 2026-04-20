#!/usr/bin/env python3
"""
Cosmos Batch v5 — process all render_output clips without cosmos version.
Longer videos: num_frames_per_chunk=93 (default), sigma_max=80.
Auto-skips already done clips.
"""
import os, json, time, glob, subprocess, urllib.request

H_RENDER_DIR = '/home/ubuntu/render_output'
H_COSMOS_OUT = '/opt/dlami/nvme/cosmos_full'
C_RENDER_DIR = '/render_output'
C_COSMOS_OUT = '/cosmos_full'

COSMOS_API = 'http://localhost:8080'
LOG_FILE   = '/tmp/cosmos_batch_full.log'
SIGMA_MAX  = 80
PROMPT = (
    'photorealistic surveillance camera footage, urban environment, '
    'security camera perspective, high quality, realistic lighting'
)

os.makedirs(H_COSMOS_OUT, exist_ok=True)

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def http_post(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def http_get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())

def wait_job(job_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            j = http_get(f'{COSMOS_API}/jobs/{job_id}')
            if j['status'] in ('done', 'error'):
                return j
        except: pass
        time.sleep(15)
    return {'status': 'timeout'}

# find clips without cosmos
all_renders = sorted(glob.glob(f'{H_RENDER_DIR}/*.mp4'))
all_renders = [r for r in all_renders if '_cosmos' not in r and '_mask' not in r]

cosmos_done = set()
for f in os.listdir(H_COSMOS_OUT):
    if f.endswith('_full.mp4'):
        cosmos_done.add(f.replace('_full.mp4', ''))

todo = [r for r in all_renders if os.path.basename(r).replace('.mp4','') not in cosmos_done]
log(f'Total raw: {len(all_renders)}, Already done: {len(cosmos_done)}, To process: {len(todo)}')

results = []
for i, h_mp4 in enumerate(todo):
    clip_id = os.path.basename(h_mp4).replace('.mp4', '')
    h_out = f'{H_COSMOS_OUT}/{clip_id}_full.mp4'
    c_input = h_mp4.replace(H_RENDER_DIR, C_RENDER_DIR)
    c_out = f'{C_COSMOS_OUT}/{clip_id}_full.mp4'

    log(f'[{i+1}/{len(todo)}] {clip_id}')

    payload = {
        'input_path': c_input,
        'output_path': c_out,
        'prompt': PROMPT,
        'edge_weight': 0.85,
        'vis_weight': 0.45,
        'multicontrol': True,
        'sigma_max': SIGMA_MAX,
        'num_frames_per_chunk': 93,
    }

    try:
        resp = http_post(f'{COSMOS_API}/transfer', payload)
        job_id = resp['job_id']
        log(f'  job_id={job_id}, waiting...')
        result = wait_job(job_id, timeout=720)
        status = result['status']
        log(f'  -> {status}')
        results.append({'clip_id': clip_id, 'status': status, 'output': h_out})
    except Exception as e:
        log(f'  ERROR: {e}')
        results.append({'clip_id': clip_id, 'status': 'error', 'error': str(e)})

with open(f'{H_COSMOS_OUT}/batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)

done_count = sum(1 for r in results if r['status'] == 'done')
log(f'BATCH COMPLETE: {done_count}/{len(todo)} done')
