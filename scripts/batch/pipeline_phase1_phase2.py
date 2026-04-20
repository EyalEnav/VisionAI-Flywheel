#!/usr/bin/env python3
"""
VisionAI Flywheel — 2-Phase Pipeline
Phase 1: Submit all render_output clips to Cosmos sigma80 (GPU1, queue-based)
Phase 2: For each completed clip → upload to VSS → query VLM → save annotation JSON
"""

import os, json, time, glob, re, subprocess, urllib.request, urllib.parse

# Host paths (for file checks, glob)
H_RENDER_DIR = '/home/ubuntu/render_output'
H_COSMOS_OUT = '/opt/dlami/nvme/cosmos_sigma80'

# Container-internal paths (sent to cosmos_api.py)
C_RENDER_DIR = '/render_output'
C_COSMOS_OUT = '/cosmos_sigma80'

COSMOS_API  = 'http://localhost:8080'
VST_BASE    = 'http://localhost:30888'
VSS_AGENT   = 'http://localhost:8000'
ANNOTATIONS = f'{H_COSMOS_OUT}/annotations.json'
LOG_FILE    = '/tmp/pipeline.log'

SIGMA_MAX = 80
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


def http_post(url, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
                                  headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def http_get(url) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def phase1_submit_all(renders):
    job_map = {}
    for h_mp4 in renders:
        clip_id = os.path.basename(h_mp4).replace('.mp4', '')
        h_out   = f'{H_COSMOS_OUT}/{clip_id}_sigma80.mp4'
        c_input = h_mp4.replace(H_RENDER_DIR, C_RENDER_DIR)
        c_out   = h_out.replace(H_COSMOS_OUT, C_COSMOS_OUT)

        if os.path.exists(h_out) and os.path.getsize(h_out) > 100_000:
            log(f'SKIP {clip_id} (already done)')
            continue

        h_mask = f'{H_RENDER_DIR}/{clip_id}_mask.mp4'
        c_mask = f'{C_RENDER_DIR}/{clip_id}_mask.mp4'
        guided_mask = c_mask if os.path.exists(h_mask) else None

        payload = {
            'input_path':  c_input,
            'output_path': c_out,
            'prompt':      PROMPT,
            'edge_weight': 0.85,
            'vis_weight':  0.45,
            'multicontrol': True,
            'sigma_max':   SIGMA_MAX,
        }
        if guided_mask:
            payload['guided_mask'] = guided_mask

        try:
            resp = http_post(f'{COSMOS_API}/transfer', payload)
            jid = resp['job_id']
            job_map[clip_id] = jid
            log(f'SUBMITTED {clip_id[:20]} → job {jid[:8]}')
        except Exception as e:
            log(f'ERROR submitting {clip_id}: {e}')
        time.sleep(0.05)

    return job_map


def upload_to_vst(mp4_path, clip_id) -> dict:
    encoded = urllib.parse.quote(clip_id.replace('-', '_')[:40])
    ts      = '2025-01-01T00%3A00%3A00.000Z'
    url     = f'{VST_BASE}/vst/api/v1/storage/file/{encoded}/{ts}'
    size    = os.path.getsize(mp4_path)
    with open(mp4_path, 'rb') as f:
        req = urllib.request.Request(
            url, data=f,
            headers={'Content-Type': 'video/mp4', 'Content-Length': str(size)},
            method='PUT'
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())


def query_vss(video_name) -> str:
    query = f'describe what is happening in {video_name}'
    payload = {'messages': [{'role': 'user', 'content': query}]}
    resp = http_post(f'{VSS_AGENT}/v1/chat/completions', payload)
    content = resp['choices'][0]['message']['content']
    content = re.sub(r'<agent-think>.*?</agent-think>', '', content, flags=re.DOTALL).strip()
    return content


def phase2_annotate(clip_id, h_out_path, annotations):
    if clip_id in annotations:
        log(f'  ANNO skip {clip_id[:20]} (already annotated)')
        return
    try:
        log(f'  UPLOAD {clip_id[:20]} → VST...')
        vst = upload_to_vst(h_out_path, clip_id)
        video_id  = vst.get('id', '')
        sensor_id = vst.get('sensorId', '')

        log(f'  QUERY VSS for {clip_id[:20]}...')
        description = query_vss(os.path.basename(h_out_path).replace('_sigma80.mp4', ''))

        annotations[clip_id] = {
            'video_filename': os.path.basename(h_out_path),
            'video_id':       video_id,
            'sensor_id':      sensor_id,
            'vss_response':   description,
            'source_render':  f'{H_RENDER_DIR}/{clip_id}.mp4',
            'cosmos_output':  h_out_path,
            'timestamp':      time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        log(f'  ✓ ANNOTATED {clip_id[:20]}: {description[:80]}...')
    except Exception as e:
        log(f'  ERROR annotating {clip_id[:20]}: {e}')


def main():
    log('=== VisionAI Pipeline START ===')
    renders = sorted(glob.glob(f'{H_RENDER_DIR}/*.mp4'))
    renders = [r for r in renders if '_mask' not in r]
    log(f'Found {len(renders)} render clips')

    if os.path.exists(ANNOTATIONS):
        with open(ANNOTATIONS) as f:
            annotations = json.load(f)
        log(f'Loaded {len(annotations)} existing annotations')
    else:
        annotations = {}

    log('=== PHASE 1: Submitting to Cosmos ===')
    job_map = phase1_submit_all(renders)
    log(f'Submitted {len(job_map)} jobs to Cosmos API')

    if not job_map:
        log('All already done — annotating any unannotated...')
        for mp4 in glob.glob(f'{H_COSMOS_OUT}/*_sigma80.mp4'):
            clip_id = os.path.basename(mp4).replace('_sigma80.mp4', '')
            phase2_annotate(clip_id, mp4, annotations)
        with open(ANNOTATIONS, 'w') as f:
            json.dump(annotations, f, indent=2)
        log(f'Done. {len(annotations)} annotations saved.')
        return

    log('=== PHASE 1+2: Polling + annotating ===')
    remaining = set(job_map.keys())
    completed = set()
    failed    = set()

    while remaining:
        time.sleep(20)
        for clip_id in list(remaining):
            jid = job_map[clip_id]
            try:
                status = http_get(f'{COSMOS_API}/jobs/{jid}')
                s = status.get('status', '?')
                if s == 'done':
                    h_out = f'{H_COSMOS_OUT}/{clip_id}_sigma80.mp4'
                    log(f'✓ Cosmos DONE: {clip_id[:20]}')
                    remaining.discard(clip_id)
                    completed.add(clip_id)
                    phase2_annotate(clip_id, h_out, annotations)
                    with open(ANNOTATIONS, 'w') as f:
                        json.dump(annotations, f, indent=2)
                elif s == 'error':
                    log(f'✗ ERROR: {clip_id[:20]} — {status.get("error","")[:80]}')
                    remaining.discard(clip_id)
                    failed.add(clip_id)
            except Exception as e:
                log(f'  poll error {clip_id[:20]}: {e}')

        log(f'  Progress: {len(completed)} done, {len(remaining)} pending, {len(failed)} failed')

    log('=== PIPELINE COMPLETE ===')
    log(f'  Completed: {len(completed)}, Failed: {len(failed)}, Annotations: {len(annotations)}')
    with open(ANNOTATIONS, 'w') as f:
        json.dump(annotations, f, indent=2)


if __name__ == '__main__':
    main()
