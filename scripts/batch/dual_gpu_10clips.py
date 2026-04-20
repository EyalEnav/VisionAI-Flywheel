#!/usr/bin/env python3
"""
Submit 10 clips to 2 Cosmos containers (5 each), poll until done.
Phase 2: Stop cosmos → Start VSS → VLM annotate all 10 clips.
"""
import os, json, time, re, glob, subprocess, urllib.request, urllib.parse, sys

H_RENDER = '/home/ubuntu/render_output'
H_OUT    = '/opt/dlami/nvme/cosmos_sigma80'
C_RENDER = '/render_output'
C_OUT    = '/cosmos_sigma80'

GPU0_API = 'http://localhost:8080'
GPU1_API = 'http://localhost:8081'
VST_BASE = 'http://localhost:30888'
VSS_AGENT= 'http://localhost:8000'
LOG_FILE = '/tmp/dual_pipeline.log'

PROMPT = ('photorealistic surveillance camera footage, urban environment, '
          'security camera perspective, high quality, realistic lighting')

os.makedirs(H_OUT, exist_ok=True)

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def http_post(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
          headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def http_get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())

def submit(api, clip_id):
    h_in  = f'{H_RENDER}/{clip_id}.mp4'
    c_in  = f'{C_RENDER}/{clip_id}.mp4'
    c_out = f'{C_OUT}/{clip_id}_sigma80.mp4'
    h_mask = f'{H_RENDER}/{clip_id}_mask.mp4'
    payload = {
        'input_path':  c_in,
        'output_path': c_out,
        'prompt':      PROMPT,
        'edge_weight': 0.85,
        'vis_weight':  0.45,
        'multicontrol': True,
        'sigma_max':   80,
    }
    if os.path.exists(h_mask):
        payload['guided_mask'] = f'{C_RENDER}/{clip_id}_mask.mp4'
    resp = http_post(f'{api}/transfer', payload)
    return resp['job_id']

# --- PHASE 1: pick 10 clips ---
renders = sorted(glob.glob(f'{H_RENDER}/*.mp4'))
renders = [r for r in renders if '_mask' not in r and '_cosmos' not in r]
clips = []
for r in renders:
    cid = os.path.basename(r).replace('.mp4','')
    out = f'{H_OUT}/{cid}_sigma80.mp4'
    if not (os.path.exists(out) and os.path.getsize(out) > 100_000):
        clips.append(cid)
    if len(clips) == 10:
        break

log(f'=== PHASE 1: {len(clips)} clips → 2 GPUs ===')

# split: odd → GPU0, even → GPU1
gpu0_clips = clips[0::2]   # indices 0,2,4,6,8
gpu1_clips = clips[1::2]   # indices 1,3,5,7,9

job_map = {}  # clip_id → (api_url, job_id)
for cid in gpu0_clips:
    jid = submit(GPU0_API, cid)
    job_map[cid] = (GPU0_API, jid)
    log(f'  GPU0 ← {cid[:16]} job={jid[:8]}')

for cid in gpu1_clips:
    jid = submit(GPU1_API, cid)
    job_map[cid] = (GPU1_API, jid)
    log(f'  GPU1 ← {cid[:16]} job={jid[:8]}')

log(f'Submitted {len(job_map)} jobs. Polling...')

# --- POLL until all done ---
remaining = set(clips)
completed = []
failed    = []

while remaining:
    time.sleep(15)
    for cid in list(remaining):
        api, jid = job_map[cid]
        try:
            s = http_get(f'{api}/jobs/{jid}')
            status = s.get('status','?')
            if status == 'done':
                log(f'✓ DONE [{api[-4:]}] {cid[:16]}')
                remaining.discard(cid)
                completed.append(cid)
            elif status == 'error':
                log(f'✗ ERROR [{api[-4:]}] {cid[:16]}: {s.get("error","")[:80]}')
                remaining.discard(cid)
                failed.append(cid)
        except Exception as e:
            log(f'  poll err {cid[:16]}: {e}')
    log(f'  done={len(completed)} remaining={len(remaining)} failed={len(failed)}')

log(f'=== Phase 1 complete: {len(completed)} OK, {len(failed)} failed ===')

# --- PHASE 2 setup: stop cosmos, start VSS ---
log('=== PHASE 2: Stopping cosmos containers ===')
subprocess.run('docker stop cosmos-gpu0 cosmos-gpu1 && docker rm cosmos-gpu0 cosmos-gpu1',
               shell=True, capture_output=True)
log('Cosmos stopped.')

log('Starting VSS stack...')
subprocess.run('cd ~/video-search-and-summarization && docker compose up -d',
               shell=True, capture_output=True)

# Wait for VSS agent to be ready
log('Waiting for VSS agent (up to 3 min)...')
for i in range(36):
    time.sleep(5)
    try:
        r = http_get(f'{VSS_AGENT}/health')
        if r.get('status') == 'ok' or r.get('ok'):
            log('VSS agent ready!')
            break
    except:
        pass
    if i % 6 == 5:
        log(f'  still waiting... ({(i+1)*5}s)')
else:
    log('WARNING: VSS agent health check failed, trying anyway...')

# --- PHASE 2: annotate completed clips ---
log('=== PHASE 2: VLM annotation ===')
annotations_file = f'{H_OUT}/annotations.json'
annotations = {}
if os.path.exists(annotations_file):
    with open(annotations_file) as f:
        annotations = json.load(f)
    log(f'Loaded {len(annotations)} existing annotations')

def upload_vst(mp4_path, clip_id):
    encoded = urllib.parse.quote(clip_id.replace('-','_')[:40])
    url = f'{VST_BASE}/vst/api/v1/storage/file/{encoded}/2025-01-01T00%3A00%3A00.000Z'
    size = os.path.getsize(mp4_path)
    with open(mp4_path, 'rb') as f:
        req = urllib.request.Request(url, data=f,
              headers={'Content-Type':'video/mp4','Content-Length':str(size)}, method='PUT')
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

def query_vss(name):
    q = f'describe what is happening in {name}'
    resp = http_post(f'{VSS_AGENT}/v1/chat/completions',
                     {'messages':[{'role':'user','content':q}]})
    content = resp['choices'][0]['message']['content']
    return re.sub(r'<agent-think>.*?</agent-think>','',content,flags=re.DOTALL).strip()

for cid in completed:
    out_path = f'{H_OUT}/{cid}_sigma80.mp4'
    if not os.path.exists(out_path):
        log(f'  SKIP {cid[:16]}: output file missing')
        continue
    try:
        log(f'  UPLOAD {cid[:16]}...')
        vst = upload_vst(out_path, cid)
        video_id  = vst.get('id','')
        sensor_id = vst.get('sensorId','')
        log(f'  QUERY VSS {cid[:16]}...')
        desc = query_vss(cid[:20])
        annotations[cid] = {
            'video_filename': os.path.basename(out_path),
            'video_id':   video_id,
            'sensor_id':  sensor_id,
            'vss_response': desc,
            'timestamp':  time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        log(f'  ✓ {cid[:16]}: {desc[:80]}')
        with open(annotations_file,'w') as f:
            json.dump(annotations, f, indent=2)
    except Exception as e:
        log(f'  ERROR {cid[:16]}: {e}')

log(f'=== ALL DONE: {len(annotations)} annotations saved to {annotations_file} ===')
