import json, requests

with open('/home/ubuntu/incident_motions.json') as f:
    motions = json.load(f)
with open('/home/ubuntu/incident_batch.json') as f:
    batch = {i['name']: i for i in json.load(f)}

for m in motions:
    m['kimodo_prompt'] = batch.get(m['name'], {}).get('kimodo_prompt', '')

RENDER_URL = 'http://localhost:9001'
CLOTHING = {
    'fall':        'grey hoodie, dark blue jeans, white sneakers',
    'injured_walk':'black jacket, dark trousers, dark shoes',
    'distress':    'casual t-shirt, jeans, sneakers',
    'violence':    'dark hoodie, black pants, boots',
    'escape':      'red jacket, dark jeans, running shoes',
    'medical':     'business shirt, khaki trousers, dress shoes',
}

render_jobs = {}
for m in motions:
    npz = m['filename']
    clothing = CLOTHING.get(m['category'], 'dark jacket, dark pants')
    resp = requests.post(RENDER_URL + '/generate',
        data={
            'prompt': clothing,
            'texture_mode': 'skin',
            'motion_file': npz,
            'fps': '30', 'width': '720', 'height': '480',
        }, timeout=30)
    jid = resp.json().get('job_id', '')
    nm = m['name']
    render_jobs[nm] = {
        'job_id': jid,
        'npz': npz,
        'category': m['category'],
        'tags': m['tags'],
        'cosmos_prompts': m['cosmos_prompts'],
        'kimodo_prompt': m['kimodo_prompt']
    }
    print('  ' + nm + '  job=' + jid[:12])

with open('/home/ubuntu/incident_render_jobs.json', 'w') as f:
    json.dump(render_jobs, f, indent=2)
print('Submitted ' + str(len(render_jobs)) + ' render jobs')
