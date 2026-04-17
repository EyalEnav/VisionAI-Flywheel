import sys
sys.path.insert(0, '/home/ubuntu/kimodo')
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch
import trimesh
import pyrender
import subprocess
import cv2
from PIL import Image, ImageDraw

NPZ_PATH  = "/tmp/kimodo_output_llama.npz"
FACE_PATH = "/tmp/user_face.jpg"
OUT_DIR   = "/tmp/face_frames"
OUT_VIDEO = "/tmp/face_surveillance.mp4"
FPS = 30
W, H = 640, 480

os.makedirs(OUT_DIR, exist_ok=True)
for f in os.listdir(OUT_DIR): os.remove(os.path.join(OUT_DIR, f))

# ── Load motion ───────────────────────────────────────────────────────────────
data = np.load(NPZ_PATH)
posed_joints    = torch.tensor(data['posed_joints'],    dtype=torch.float32)
global_rot_mats = torch.tensor(data['global_rot_mats'], dtype=torch.float32)
T = posed_joints.shape[0]
min_y = posed_joints[:,:,1].min().item()
posed_joints[:,:,1] -= min_y

# ── Load SOMASkin ─────────────────────────────────────────────────────────────
from kimodo.skeleton import SOMASkeleton77
from kimodo.viz.soma_skin import SOMASkin

skel_path = '/home/ubuntu/kimodo/kimodo/assets/skeletons/somaskel77'
skin_npz  = np.load(f'{skel_path}/skin_standard.npz', allow_pickle=True)
skeleton  = SOMASkeleton77(skel_path)
bind_rig  = torch.tensor(skin_npz['bind_rig_transform'], dtype=torch.float32)
skeleton.neutral_joints = bind_rig[:, :3, 3].clone()
soma_skin = SOMASkin(skeleton)

bind_verts = skin_npz['bind_vertices']  # [V,3]
V = len(bind_verts)

# ── Face image: crop just the face region ─────────────────────────────────────
face_img = cv2.imread(FACE_PATH)
face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

# Use insightface to detect face bounding box (mediapipe replaced)
try:
    import insightface
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(allowed_modules=['detection'], providers=['CPUExecutionProvider'])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    faces = fa.get(face_rgb)
    if faces:
        bbox = faces[0].bbox.astype(int)
        ih, iw = face_rgb.shape[:2]
        x1 = max(0, bbox[0] - 20)
        y1 = max(0, bbox[1] - 40)
        x2 = min(iw, bbox[2] + 20)
        y2 = min(ih, bbox[3] + 20)
        face_crop = face_rgb[y1:y2, x1:x2]
        print(f"Face detected (insightface): {x1},{y1} -> {x2},{y2}")
    else:
        raise ValueError("No face detected")
except Exception as e:
    print(f"insightface fallback: {e} — using center crop")
    ih, iw = face_rgb.shape[:2]
    face_crop = face_rgb[int(ih*0.05):int(ih*0.65), int(iw*0.1):int(iw*0.9)]
    print("Using center crop fallback")

face_crop = cv2.resize(face_crop, (256, 256))
face_pil  = Image.fromarray(face_crop)

# ── Vertex coloring with face texture on head zone ────────────────────────────
y_min_b, y_max_b = bind_verts[:,1].min(), bind_verts[:,1].max()
y_norm = (bind_verts[:,1] - y_min_b) / (y_max_b - y_min_b)
x_abs  = np.abs(bind_verts[:,0])
z_vals = bind_verts[:,2]

vertex_colors = np.zeros((V, 4), dtype=np.uint8)

SHOE    = np.array([35,  30,  25, 255], dtype=np.uint8)
SOCK    = np.array([220, 220, 220, 255], dtype=np.uint8)
TROUSER = np.array([28,  38,  70, 255], dtype=np.uint8)
BELT    = np.array([20,  15,  10, 255], dtype=np.uint8)
SHIRT   = np.array([90,  105, 120, 255], dtype=np.uint8)
SLEEVE  = np.array([90,  105, 120, 255], dtype=np.uint8)
SKIN_C  = np.array([210, 175, 140, 255], dtype=np.uint8)

face_arr = np.array(face_pil)  # [256,256,3]

# Head zone: y_norm > 0.82
head_mask = y_norm > 0.82
head_verts = bind_verts[head_mask]  # subset

if head_verts.shape[0] > 0:
    # Project head verts onto face image using x,y normalised coords
    # x → horizontal, y → vertical (inverted), z front-back weight
    hx = head_verts[:,0]
    hy = head_verts[:,1]
    hz = head_verts[:,2]

    hx_norm = (hx - hx.min()) / (hx.max() - hx.min() + 1e-6)  # [0..1] left→right
    hy_norm = 1.0 - (hy - hy.min()) / (hy.max() - hy.min() + 1e-6)  # [0..1] top→bottom

    # Front-back weight: vertices at front (higher z) get full face texture
    # vertices at back get hair/dark color
    hz_norm = (hz - hz.min()) / (hz.max() - hz.min() + 1e-6)  # 0=back, 1=front

    head_indices = np.where(head_mask)[0]
    for idx, (ux, uy, uz) in zip(head_indices, zip(hx_norm, hy_norm, hz_norm)):
        if uz > 0.35:  # front hemisphere → face texture
            px = int(np.clip(ux * 255, 0, 255))
            py = int(np.clip(uy * 255, 0, 255))
            r, g, b = face_arr[py, px]
            vertex_colors[idx] = [r, g, b, 255]
        else:  # back of head → dark brown (hair)
            vertex_colors[idx] = [45, 30, 20, 255]

# Body zones
for v in range(V):
    if head_mask[v]: continue  # already set
    yn = y_norm[v]; xn = x_abs[v]
    if yn < 0.08:              vertex_colors[v] = SHOE
    elif yn < 0.14:            vertex_colors[v] = SOCK
    elif yn < 0.52:            vertex_colors[v] = TROUSER
    elif yn < 0.56:            vertex_colors[v] = BELT
    elif yn < 0.80:
        vertex_colors[v] = SLEEVE if xn > 0.18 else SHIRT
    else:                      vertex_colors[v] = SKIN_C

# Hand skin
hand_mask = (y_norm > 0.20) & (y_norm < 0.60) & (x_abs > 0.22)
vertex_colors[hand_mask] = SKIN_C

print(f"Head verts: {head_mask.sum()}, body done")

# ── Pyrender ──────────────────────────────────────────────────────────────────
renderer = pyrender.OffscreenRenderer(W, H)
camera   = pyrender.PerspectiveCamera(yfov=np.pi/3.5, aspectRatio=W/H)

def add_dir_light(scene, direction, color, intensity):
    lgt = pyrender.DirectionalLight(color=color, intensity=intensity)
    d = np.array(direction, dtype=float); d /= np.linalg.norm(d)
    up = np.array([0,1,0]) if abs(d[1])<0.99 else np.array([1,0,0])
    z = -d; x = np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    p = np.eye(4); p[:3,0]=x; p[:3,1]=y; p[:3,2]=z; p[:3,3]=d*5
    scene.add(lgt, pose=p)

print(f"Rendering {T} frames...")
for i in range(T):
    j_pos = posed_joints[i:i+1]
    j_rot = global_rot_mats[i:i+1]
    verts_t  = soma_skin.skin(j_rot, j_pos, rot_is_global=True)
    verts_np = verts_t[0].detach().numpy()
    faces_np = soma_skin.faces.detach().numpy()

    mesh_tri = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    mesh_tri.visual = trimesh.visual.ColorVisuals(mesh=mesh_tri, vertex_colors=vertex_colors)
    mesh_pr = pyrender.Mesh.from_trimesh(mesh_tri, smooth=True)

    ground = trimesh.creation.box(extents=[5,0.02,5])
    ground.apply_translation([0,-0.01,0])
    ground.visual = trimesh.visual.ColorVisuals(
        mesh=ground, vertex_colors=np.tile([75,75,75,255],(len(ground.vertices),1)).astype(np.uint8))
    ground_pr = pyrender.Mesh.from_trimesh(ground)

    scene = pyrender.Scene(ambient_light=[0.3,0.3,0.3], bg_color=[15,15,15,255])
    scene.add(mesh_pr); scene.add(ground_pr)
    add_dir_light(scene, [-1,-2,-1], [1.0,0.95,0.85], 6.0)
    add_dir_light(scene, [ 1,-1, 1], [0.5,0.6, 0.8], 3.0)
    add_dir_light(scene, [ 0, 1, 0], [0.8,0.8, 1.0], 2.0)

    cx = verts_np[:,0].mean(); cz = verts_np[:,2].mean()
    cam_pos = np.array([cx+1.8, 4.8, cz+3.8])
    target  = np.array([cx, 0.9, cz])
    up      = np.array([0,1,0])
    z_ax = cam_pos-target; z_ax/=np.linalg.norm(z_ax)
    x_ax = np.cross(up,z_ax); x_ax/=np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax,x_ax)
    cp = np.eye(4); cp[:3,0]=x_ax; cp[:3,1]=y_ax; cp[:3,2]=z_ax; cp[:3,3]=cam_pos
    scene.add(camera, pose=cp)

    color_img, _ = renderer.render(scene)
    img  = Image.fromarray(color_img)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0,0,W,22], fill=(0,0,0,200))
    draw.text((8,4),  f"■ REC  CAM-04  {i/FPS:05.2f}s", fill=(255,50,50))
    draw.text((W-170,4), "AI SURVEILLANCE ACTIVE", fill=(255,180,0))
    draw.rectangle([0,H-20,W,H], fill=(0,0,0,180))
    draw.text((8,H-16), "CROWD PUSH INCIDENT — PERSON DETECTED", fill=(80,255,100))
    from datetime import datetime, timedelta
    ts = datetime(2026,4,12,19,0,0) + timedelta(seconds=i/FPS)
    draw.text((W-165,H-16), ts.strftime("%Y-%m-%d  %H:%M:%S"), fill=(180,180,180))

    img.save(f"{OUT_DIR}/frame_{i:04d}.png")
    if i % 30 == 0: print(f"  frame {i}/{T}")

renderer.delete()
print("Encoding...")
subprocess.run([
    "ffmpeg","-y","-framerate",str(FPS),
    "-i",f"{OUT_DIR}/frame_%04d.png",
    "-vf","scale=640:480",
    "-c:v","libx264","-pix_fmt","yuv420p","-crf","18",
    OUT_VIDEO
], check=True, capture_output=True)
print(f"Done: {os.path.getsize(OUT_VIDEO)//1024} KB")
