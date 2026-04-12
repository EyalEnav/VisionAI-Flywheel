"""
Cosmos-Texture Render Pipeline
1. Send clothing/texture prompt to cosmos-reason2-8b
2. Parse RGB colors for each body zone
3. Apply to SOMA mesh
4. Optional: insightface swap on head
5. Render video
"""
import sys, os, json, re, uuid, subprocess
sys.path.insert(0, '/home/ubuntu/kimodo')
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch
import trimesh
import pyrender
import cv2
from PIL import Image, ImageDraw
from datetime import datetime, timedelta
import urllib.request, urllib.parse
import http.client

COSMOS_URL = "http://localhost:30082/v1/chat/completions"
COSMOS_MODEL = "nvidia/cosmos-reason2-8b"

def cosmos_chat(prompt, max_tokens=500):
    payload = json.dumps({
        "model": COSMOS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }).encode()
    conn = http.client.HTTPConnection("localhost", 30082, timeout=60)
    conn.request("POST", "/v1/chat/completions",
                 body=payload,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data["choices"][0]["message"]["content"]

def parse_rgb(text, label):
    """Extract RGB for a zone label from Cosmos output"""
    patterns = [
        rf'{label}[^\n]*?RGB[:\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)',
        rf'RGB[:\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)[^\n]*{label}',
        rf'\*\*{label}[^*]*\*\*[^\n]*\n[^\n]*RGB[:\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            # sanity check
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                return np.array([r, g, b, 255], dtype=np.uint8)
    return None

def generate_texture_from_cosmos(texture_prompt):
    """Ask Cosmos for structured texture colors"""
    full_prompt = f"""You are a 3D texture artist for realistic human figures.
Clothing description: "{texture_prompt}"

Output ONLY 7 lines, no explanations. Use realistic RGB values for a real human:

Torso RGB: R, G, B
Legs RGB: R, G, B
Shoes RGB: R, G, B
Skin RGB: 185, 140, 100
Hair RGB: 30, 20, 15
Belt RGB: R, G, B
Socks RGB: R, G, B

Rules:
- Skin must always be a realistic human tone like 185,140,100 or 210,170,130 (NEVER white 255,255,255)
- Hair must be dark (black/brown) unless specified otherwise
- Colors must match the clothing description exactly
- Output ONLY the 7 lines above with realistic values"""

    print(f"Querying Cosmos-Reason2 for texture: '{texture_prompt}'")
    response = cosmos_chat(full_prompt, max_tokens=200)
    print(f"Cosmos response:\n{response}\n")

    defaults = {
        "Torso":  np.array([90, 105, 120, 255], dtype=np.uint8),
        "Legs":   np.array([28, 38, 70, 255],   dtype=np.uint8),
        "Shoes":  np.array([35, 30, 25, 255],   dtype=np.uint8),
        "Skin":   np.array([195,155,120,255],   dtype=np.uint8),
        "Hair":   np.array([30, 20, 15, 255],   dtype=np.uint8),
        "Belt":   np.array([20, 15, 10, 255],   dtype=np.uint8),
        "Socks":  np.array([220,220,220,255],   dtype=np.uint8),
    }

    colors = {}
    for zone, default in defaults.items():
        c = parse_rgb(response, zone)
        if c is not None:
            print(f"  {zone}: RGB{tuple(c[:3])}")
            colors[zone] = c
        else:
            print(f"  {zone}: fallback to default {tuple(default[:3])}")
            colors[zone] = default

    return colors, response

def apply_cosmos_colors(bind_verts, colors):
    V = len(bind_verts)
    vertex_colors = np.zeros((V, 4), dtype=np.uint8)
    y_min_b = bind_verts[:,1].min()
    y_max_b = bind_verts[:,1].max()
    y_norm  = (bind_verts[:,1] - y_min_b) / (y_max_b - y_min_b)
    x_abs   = np.abs(bind_verts[:,0])
    z_vals  = bind_verts[:,2]

    head_mask = y_norm > 0.80
    neck_mask = (y_norm > 0.74) & (y_norm <= 0.80)
    z_min = bind_verts[head_mask, 2].min() if head_mask.sum() > 0 else 0
    z_max = bind_verts[head_mask, 2].max() if head_mask.sum() > 0 else 1

    for v in range(V):
        yn = y_norm[v]; xn = x_abs[v]; zn = z_vals[v]
        if head_mask[v]:
            # front of head = skin, back = hair
            hz_norm = (zn - z_min) / max(z_max - z_min, 1e-6)
            if hz_norm > 0.4:
                vertex_colors[v] = colors["Skin"]
            else:
                vertex_colors[v] = colors["Hair"]
        elif neck_mask[v]:
            vertex_colors[v] = colors["Skin"]
        elif yn < 0.08:
            vertex_colors[v] = colors["Shoes"]
        elif yn < 0.14:
            vertex_colors[v] = colors["Socks"]
        elif yn < 0.52:
            vertex_colors[v] = colors["Legs"]
        elif yn < 0.56:
            vertex_colors[v] = colors["Belt"]
        elif yn < 0.80:
            vertex_colors[v] = colors["Torso"] if xn < 0.18 else colors["Skin"]
        else:
            vertex_colors[v] = colors["Skin"]

    # Hands
    hand_mask = (y_norm > 0.20) & (y_norm < 0.60) & (x_abs > 0.22)
    vertex_colors[hand_mask] = colors["Skin"]

    return vertex_colors

def render_with_cosmos_texture(
    texture_prompt,
    npz_path="/tmp/kimodo_output_llama.npz",
    face_path=None,
    out_video="/tmp/cosmos_render.mp4",
    fps=30, W=640, H=480
):
    # 1. Get colors from Cosmos
    colors, cosmos_response = generate_texture_from_cosmos(texture_prompt)

    # 2. Load motion
    data = np.load(npz_path)
    posed_joints    = torch.tensor(data['posed_joints'],    dtype=torch.float32)
    global_rot_mats = torch.tensor(data['global_rot_mats'], dtype=torch.float32)
    T = posed_joints.shape[0]
    min_y = posed_joints[:,:,1].min().item()
    posed_joints[:,:,1] -= min_y

    # 3. Load SOMASkin
    from kimodo.skeleton import SOMASkeleton77
    from kimodo.viz.soma_skin import SOMASkin
    skel_path = '/home/ubuntu/kimodo/kimodo/assets/skeletons/somaskel77'
    skin_npz  = np.load(f'{skel_path}/skin_standard.npz', allow_pickle=True)
    skeleton  = SOMASkeleton77(skel_path)
    bind_rig  = torch.tensor(skin_npz['bind_rig_transform'], dtype=torch.float32)
    skeleton.neutral_joints = bind_rig[:, :3, 3].clone()
    soma_skin = SOMASkin(skeleton)
    bind_verts = skin_npz['bind_vertices']
    V = len(bind_verts)

    # 4. Apply Cosmos texture
    vertex_colors = apply_cosmos_colors(bind_verts, colors)
    print(f"Applied Cosmos texture colors to {V} vertices")

    # 5. Face swap setup
    do_faceswap = (face_path and os.path.exists(face_path))
    src_face = None; swapper = None; face_app = None
    if do_faceswap:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
        face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640,640))
        swapper = get_model('/tmp/insightface_models/inswapper_128.onnx', download=False,
                            providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        src_img = cv2.imread(face_path)
        src_faces = face_app.get(src_img)
        if src_faces:
            src_face = sorted(src_faces, key=lambda x: x.bbox[2]*x.bbox[3], reverse=True)[0]
            print("Face embedding loaded for swap")

    # 6. Render
    out_dir = "/tmp/cosmos_frames"
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir): os.remove(os.path.join(out_dir, f))

    renderer = pyrender.OffscreenRenderer(W, H)
    camera   = pyrender.PerspectiveCamera(yfov=np.pi/3.5, aspectRatio=W/H)

    def add_light(scene, d, c, i):
        lgt = pyrender.DirectionalLight(color=c, intensity=i)
        d = np.array(d, dtype=float); d /= np.linalg.norm(d)
        up = np.array([0,1,0]) if abs(d[1])<0.99 else np.array([1,0,0])
        z=-d; x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
        p=np.eye(4); p[:3,0]=x; p[:3,1]=y; p[:3,2]=z; p[:3,3]=d*5
        scene.add(lgt, pose=p)

    swap_count = 0
    print(f"Rendering {T} frames...")
    for i in range(T):
        verts_t  = soma_skin.skin(global_rot_mats[i:i+1], posed_joints[i:i+1], rot_is_global=True)
        verts_np = verts_t[0].detach().numpy()

        mesh_tri = trimesh.Trimesh(vertices=verts_np, faces=soma_skin.faces.detach().numpy(), process=False)
        mesh_tri.visual = trimesh.visual.ColorVisuals(mesh=mesh_tri, vertex_colors=vertex_colors)

        ground = trimesh.creation.box(extents=[5,0.02,5])
        ground.apply_translation([0,-0.01,0])
        ground.visual = trimesh.visual.ColorVisuals(mesh=ground,
            vertex_colors=np.tile([55,55,55,255],(len(ground.vertices),1)).astype(np.uint8))

        scene = pyrender.Scene(ambient_light=[0.3,0.3,0.3], bg_color=[18,18,18,255])
        scene.add(pyrender.Mesh.from_trimesh(mesh_tri, smooth=True))
        scene.add(pyrender.Mesh.from_trimesh(ground))
        add_light(scene, [-1,-2,-1], [1.0,0.95,0.85], 6.0)
        add_light(scene, [ 1,-1, 1], [0.5,0.6, 0.8], 3.0)
        add_light(scene, [ 0, 1, 0], [0.8,0.8, 1.0], 2.0)

        cx=verts_np[:,0].mean(); cz=verts_np[:,2].mean()
        cam_pos=np.array([cx+1.8,4.8,cz+3.8]); target=np.array([cx,0.9,cz])
        z_ax=cam_pos-target; z_ax/=np.linalg.norm(z_ax)
        x_ax=np.cross([0,1,0],z_ax); x_ax/=np.linalg.norm(x_ax)
        y_ax=np.cross(z_ax,x_ax)
        cp=np.eye(4); cp[:3,0]=x_ax; cp[:3,1]=y_ax; cp[:3,2]=z_ax; cp[:3,3]=cam_pos
        scene.add(camera, pose=cp)

        color_img, _ = renderer.render(scene)
        frame_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        if do_faceswap and src_face:
            dst_faces = face_app.get(frame_bgr)
            if dst_faces:
                dst = sorted(dst_faces, key=lambda x: x.bbox[2]*x.bbox[3], reverse=True)[0]
                frame_bgr = swapper.get(frame_bgr, dst, src_face, paste_back=True)
                swap_count += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0,0,W,22], fill=(0,0,0,200))
        draw.text((8,4), f"■ REC  CAM-04  {i/fps:05.2f}s", fill=(255,50,50))
        draw.text((W-200,4), f"COSMOS: {texture_prompt[:22]}", fill=(100,255,200))
        draw.rectangle([0,H-20,W,H], fill=(0,0,0,180))
        ts = datetime(2026,4,12,19,0,0)+timedelta(seconds=i/fps)
        draw.text((8,H-16), "SYNTHETIC VLM DATASET", fill=(80,255,100))
        draw.text((W-165,H-16), ts.strftime("%Y-%m-%d  %H:%M:%S"), fill=(180,180,180))
        img.save(f"{out_dir}/frame_{i:04d}.png")
        if i % 30 == 0: print(f"  frame {i}/{T}  swaps={swap_count}")

    renderer.delete()
    print("Encoding...")
    subprocess.run(["ffmpeg","-y","-framerate",str(fps),
        "-i",f"{out_dir}/frame_%04d.png",
        "-vf",f"scale={W}:{H}","-c:v","libx264","-pix_fmt","yuv420p","-crf","18",
        out_video], check=True, capture_output=True)
    size = os.path.getsize(out_video)//1024
    print(f"Done: {size} KB → {out_video}")
    return colors, cosmos_response, out_video

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "orange construction worker vest, khaki cargo pants, white hard hat, brown work boots"
    face_path = "/tmp/user_face.jpg" if os.path.exists("/tmp/user_face.jpg") else None
    colors, cosmos_resp, vid = render_with_cosmos_texture(
        texture_prompt=prompt,
        face_path=face_path,
        out_video="/tmp/cosmos_render.mp4"
    )
    print("\nFinal colors:")
    for k, v in colors.items():
        print(f"  {k}: RGB{tuple(v[:3])}")
