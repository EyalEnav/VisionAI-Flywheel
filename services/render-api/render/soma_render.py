"""
SOMA mesh renderer — Multi-character crowd support.
texture_mode:
  "cosmos"   → colored clothing zones (colors dict required)
  "skeleton" → flat grey skeleton only (for Cosmos Transfer input)
  "faceswap" → cosmos colors + InsightFace identity swap on head

crowd_extras: list of dicts:
  { npz_path, colors, offset_x, offset_z, time_offset, scale }
"""
import os, subprocess
import numpy as np
import torch
import trimesh
import pyrender
import cv2
from PIL import Image, ImageDraw
from datetime import datetime, timedelta
import random

KIMODO_PATH = os.environ.get("KIMODO_PATH", "/kimodo")
SKEL_PATH   = f"{KIMODO_PATH}/kimodo/assets/skeletons/somaskel77"
INSIGHTFACE_MODEL = os.environ.get("INSIGHTFACE_MODEL", "/models/inswapper_128.onnx")


def _load_soma(npz_path):
    import sys; sys.path.insert(0, KIMODO_PATH)
    from kimodo.skeleton import SOMASkeleton77
    import importlib.util, sys
    _spec = importlib.util.spec_from_file_location("soma_skin", "/kimodo/kimodo/viz/soma_skin.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    SOMASkin = _mod.SOMASkin

    skin_npz  = np.load(f"{SKEL_PATH}/skin_standard.npz", allow_pickle=True)
    skeleton  = SOMASkeleton77(SKEL_PATH)
    bind_rig  = torch.tensor(skin_npz["bind_rig_transform"], dtype=torch.float32)
    skeleton.neutral_joints = bind_rig[:, :3, 3].clone()
    soma_skin = SOMASkin(skeleton)
    bind_verts = skin_npz["bind_vertices"]

    data = np.load(npz_path)
    pj = data["posed_joints"]
    gr = data["global_rot_mats"]
    # squeeze batch dimension if present (1, T, J, 3) -> (T, J, 3)
    if pj.ndim == 4: pj = pj[0]
    if gr.ndim == 5: gr = gr[0]
    posed_joints    = torch.tensor(pj, dtype=torch.float32)
    global_rot_mats = torch.tensor(gr, dtype=torch.float32)
    min_y = posed_joints[:,:,1].min().item()
    posed_joints[:,:,1] -= min_y

    return soma_skin, bind_verts, posed_joints, global_rot_mats


def _vertex_colors_cosmos(bind_verts, colors):
    V = len(bind_verts)
    vc = np.zeros((V, 4), dtype=np.uint8)
    y_min, y_max = bind_verts[:,1].min(), bind_verts[:,1].max()
    y_norm = (bind_verts[:,1] - y_min) / (y_max - y_min)
    x_abs  = np.abs(bind_verts[:,0])
    z_vals = bind_verts[:,2]

    head_mask = y_norm > 0.80
    z_min = bind_verts[head_mask,2].min() if head_mask.sum() > 0 else 0
    z_max = bind_verts[head_mask,2].max() if head_mask.sum() > 0 else 1

    for v in range(V):
        yn, xn, zn = y_norm[v], x_abs[v], z_vals[v]
        if head_mask[v]:
            hz = (zn - z_min) / max(z_max - z_min, 1e-6)
            vc[v] = colors["Skin"] if hz > 0.4 else colors["Hair"]
        elif (y_norm[v] > 0.74) and not head_mask[v]:
            vc[v] = colors["Skin"]
        elif yn < 0.08:   vc[v] = colors["Shoes"]
        elif yn < 0.14:   vc[v] = colors["Socks"]
        elif yn < 0.52:   vc[v] = colors["Legs"]
        elif yn < 0.56:   vc[v] = colors["Belt"]
        elif yn < 0.80:   vc[v] = colors["Torso"] if xn < 0.18 else colors["Skin"]
        else:             vc[v] = colors["Skin"]

    vc[(y_norm > 0.20) & (y_norm < 0.60) & (x_abs > 0.22)] = colors["Skin"]
    return vc


def _vertex_colors_skeleton(bind_verts):
    V = len(bind_verts)
    vc = np.full((V, 4), [80, 80, 90, 255], dtype=np.uint8)
    return vc


def _add_light(scene, d, c, i):
    lgt = pyrender.DirectionalLight(color=c, intensity=i)
    d = np.array(d, dtype=float); d /= np.linalg.norm(d)
    up = np.array([0,1,0]) if abs(d[1]) < 0.99 else np.array([1,0,0])
    z=-d; x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    p=np.eye(4); p[:3,0]=x; p[:3,1]=y; p[:3,2]=z; p[:3,3]=d*5
    scene.add(lgt, pose=p)


# ─── Crowd palette pool ───────────────────────────────────────────
CROWD_PALETTES = [
    {"Torso":[220,50,50,255],  "Legs":[40,40,120,255],  "Shoes":[20,20,20,255],  "Socks":[240,240,240,255], "Belt":[30,20,10,255],  "Skin":[210,170,130,255], "Hair":[30,20,10,255]},
    {"Torso":[50,120,220,255], "Legs":[80,80,80,255],   "Shoes":[50,30,10,255],  "Socks":[200,200,200,255], "Belt":[60,60,60,255],  "Skin":[240,200,160,255], "Hair":[80,50,20,255]},
    {"Torso":[30,160,60,255],  "Legs":[20,20,80,255],   "Shoes":[30,30,30,255],  "Socks":[240,240,240,255], "Belt":[40,30,10,255],  "Skin":[180,130,90,255],  "Hair":[10,10,10,255]},
    {"Torso":[200,200,50,255], "Legs":[60,30,10,255],   "Shoes":[20,20,20,255],  "Socks":[230,230,230,255], "Belt":[40,40,40,255],  "Skin":[255,220,180,255], "Hair":[180,140,80,255]},
    {"Torso":[180,80,180,255], "Legs":[30,30,30,255],   "Shoes":[60,40,20,255],  "Socks":[240,240,240,255], "Belt":[50,50,50,255],  "Skin":[200,160,120,255], "Hair":[60,40,20,255]},
    {"Torso":[240,120,30,255], "Legs":[50,50,100,255],  "Shoes":[25,25,25,255],  "Socks":[220,220,220,255], "Belt":[35,25,10,255],  "Skin":[160,110,70,255],  "Hair":[20,15,10,255]},
    {"Torso":[60,60,60,255],   "Legs":[200,50,50,255],  "Shoes":[40,40,40,255],  "Socks":[200,200,200,255], "Belt":[30,30,30,255],  "Skin":[220,180,140,255], "Hair":[100,70,40,255]},
    {"Torso":[255,255,255,255],"Legs":[100,100,100,255],"Shoes":[30,30,30,255],  "Socks":[230,230,230,255], "Belt":[50,50,50,255],  "Skin":[245,210,175,255], "Hair":[50,30,10,255]},
]


def _get_crowd_mesh(soma_skin, bind_verts, posed_joints, global_rot_mats,
                    frame_idx, time_offset, offset_x, offset_z, colors, texture_mode):
    """Return a trimesh for one extra character at a given frame."""
    T = posed_joints.shape[0]
    t = (frame_idx + time_offset) % T  # loop motion with time offset

    verts_t  = soma_skin.skin(global_rot_mats[t:t+1], posed_joints[t:t+1], rot_is_global=True)
    verts_np = verts_t[0].detach().numpy().copy()

    # Apply XZ position offset (keep Y = ground-level)
    verts_np[:, 0] += offset_x
    verts_np[:, 2] += offset_z

    # Random facing direction (yaw rotation around Y axis)
    # Already encoded in npz, just offset position

    if texture_mode == "skeleton":
        vc = _vertex_colors_skeleton(bind_verts)
    else:
        vc = _vertex_colors_cosmos(bind_verts, colors)

    mesh_tri = trimesh.Trimesh(vertices=verts_np, faces=soma_skin.faces.detach().numpy(), process=False)
    mesh_tri.visual = trimesh.visual.ColorVisuals(mesh=mesh_tri, vertex_colors=vc)
    return mesh_tri


def render(npz_path, out_video, texture_mode="cosmos", colors=None,
           face_path=None, fps=30, W=640, H=480,
           crowd_extras=None):
    """
    crowd_extras: list of dicts with keys:
        npz_path    - motion file for this extra
        colors      - palette dict (or None → random from pool)
        offset_x    - X world offset in meters
        offset_z    - Z world offset in meters
        time_offset - frame offset (int) for motion loop
    """
    soma_skin, bind_verts, posed_joints, global_rot_mats = _load_soma(npz_path)
    T = posed_joints.shape[0]

    # Pre-load all extras
    extras_data = []
    if crowd_extras:
        for ex in crowd_extras:
            ex_skin, ex_bverts, ex_pj, ex_grm = _load_soma(ex["npz_path"])
            palette = ex.get("colors") or random.choice(CROWD_PALETTES)
            extras_data.append({
                "soma_skin": ex_skin,
                "bind_verts": ex_bverts,
                "posed_joints": ex_pj,
                "global_rot_mats": ex_grm,
                "colors": palette,
                "offset_x": ex.get("offset_x", 0.0),
                "offset_z": ex.get("offset_z", 0.0),
                "time_offset": ex.get("time_offset", 0),
            })

    if texture_mode == "skeleton":
        vertex_colors = _vertex_colors_skeleton(bind_verts)
    else:
        vertex_colors = _vertex_colors_cosmos(bind_verts, colors)

    # InsightFace setup
    swapper = src_face = face_app = None
    if texture_mode == "faceswap" and face_path and os.path.exists(face_path) and os.path.exists(INSIGHTFACE_MODEL):
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
        face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        swapper = get_model(INSIGHTFACE_MODEL, download=False,
                            providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        src_img = cv2.imread(face_path)
        src_faces = face_app.get(src_img)
        if src_faces:
            src_face = sorted(src_faces, key=lambda x: x.bbox[2]*x.bbox[3], reverse=True)[0]

    out_dir = out_video.replace(".mp4", "_frames")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir): os.remove(os.path.join(out_dir, f))

    renderer = pyrender.OffscreenRenderer(W, H)
    camera   = pyrender.PerspectiveCamera(yfov=np.pi/3.5, aspectRatio=W/H)

    # ── Compute root trajectory for camera placement ──
    # Extract XZ path across all frames to decide camera mode
    root_pos_all = []
    for _i in range(T):
        _vt = soma_skin.skin(global_rot_mats[_i:_i+1], posed_joints[_i:_i+1], rot_is_global=True)
        _vn = _vt[0].detach().numpy()
        root_pos_all.append([_vn[:,0].mean(), _vn[:,2].mean()])
    root_pos_all = np.array(root_pos_all)  # (T, 2) — XZ centers per frame

    # Total XZ displacement
    xz_total = root_pos_all[-1] - root_pos_all[0]
    xz_dist  = np.linalg.norm(xz_total)

    # Camera mode: if person travels > 1.5m total → tracking side-cam, else fixed overhead
    TRAVEL_THRESHOLD = 1.5
    use_travel_cam = xz_dist > TRAVEL_THRESHOLD

    if use_travel_cam:
        # Side-view camera: positioned perpendicular to travel direction
        travel_dir = xz_total / (xz_dist + 1e-6)          # normalized direction in XZ
        perp_dir   = np.array([-travel_dir[1], travel_dir[0]])  # 90° perpendicular in XZ

        # Camera sits to the side, slightly behind midpoint of journey
        mid_xz = root_pos_all[T//2]
        cam_side_dist = 5.0   # meters to the side
        cam_height    = 3.5   # meters up
        cam_xz = mid_xz + perp_dir * cam_side_dist
        cam_pos_fixed = np.array([cam_xz[0], cam_height, cam_xz[1]])
        cam_target_fixed = np.array([mid_xz[0], 1.0, mid_xz[1]])

    for i in range(T):
        # ── Protagonist ──
        verts_t  = soma_skin.skin(global_rot_mats[i:i+1], posed_joints[i:i+1], rot_is_global=True)
        verts_np = verts_t[0].detach().numpy()
        cx = verts_np[:,0].mean(); cz = verts_np[:,2].mean()

        mesh_tri = trimesh.Trimesh(vertices=verts_np, faces=soma_skin.faces.detach().numpy(), process=False)
        mesh_tri.visual = trimesh.visual.ColorVisuals(mesh=mesh_tri, vertex_colors=vertex_colors)

        # ── Scene with ground plane ──
        bg = [18, 18, 18, 255] if texture_mode != "skeleton" else [0, 0, 0, 255]
        scene = pyrender.Scene(ambient_light=[0.3,0.3,0.3], bg_color=bg)
        scene.add(pyrender.Mesh.from_trimesh(mesh_tri, smooth=True))

        # ── Ground plane: 8x8m asphalt slab centered under character ──
        # Y = foot level (min Y of protagonist verts, slightly below)
        foot_y = float(verts_np[:, 1].min()) - 0.01
        gw = 8.0  # width/depth in meters
        gverts = np.array([
            [cx - gw, foot_y, cz - gw],
            [cx + gw, foot_y, cz - gw],
            [cx + gw, foot_y, cz + gw],
            [cx - gw, foot_y, cz + gw],
        ], dtype=np.float32)
        gfaces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        # Asphalt color: dark grey with slight blue-tone
        if texture_mode == "skeleton":
            gc = np.array([[40, 40, 40, 255]]*4, dtype=np.uint8)
        else:
            gc = np.array([[55, 55, 60, 255]]*4, dtype=np.uint8)
        ground_tri = trimesh.Trimesh(vertices=gverts, faces=gfaces, process=False)
        ground_tri.visual = trimesh.visual.ColorVisuals(mesh=ground_tri, vertex_colors=gc)
        scene.add(pyrender.Mesh.from_trimesh(ground_tri, smooth=False))

        # ── Crowd extras ──
        for ex in extras_data:
            ex_mesh = _get_crowd_mesh(
                ex["soma_skin"], ex["bind_verts"],
                ex["posed_joints"], ex["global_rot_mats"],
                i, ex["time_offset"],
                ex["offset_x"], ex["offset_z"],
                ex["colors"], texture_mode
            )
            scene.add(pyrender.Mesh.from_trimesh(ex_mesh, smooth=True))

        # ── Lighting ──
        _add_light(scene, [-1,-2,-1], [1.0,0.95,0.85], 6.0)
        _add_light(scene, [ 1,-1, 1], [0.5,0.6, 0.8], 3.0)
        _add_light(scene, [ 0, 1, 0], [0.8,0.8, 1.0], 2.0)

        # ── Camera: side-view for traveling motion, overhead for in-place ──
        if use_travel_cam:
            cam_pos = cam_pos_fixed
            target  = cam_target_fixed
        else:
            # Classic surveillance overhead tracking
            cam_pos = np.array([cx+1.8, 4.8, cz+3.8])
            target  = np.array([cx, 0.9, cz])
        z_ax = cam_pos-target; z_ax /= np.linalg.norm(z_ax)
        x_ax = np.cross([0,1,0], z_ax); x_ax /= np.linalg.norm(x_ax)
        y_ax = np.cross(z_ax, x_ax)
        cp = np.eye(4); cp[:3,0]=x_ax; cp[:3,1]=y_ax; cp[:3,2]=z_ax; cp[:3,3]=cam_pos
        scene.add(camera, pose=cp)

        color_img, _ = renderer.render(scene)
        frame_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        # Face swap on protagonist only
        if texture_mode == "faceswap" and src_face and face_app:
            dst_faces = face_app.get(frame_bgr)
            if dst_faces:
                dst = sorted(dst_faces, key=lambda x: x.bbox[2]*x.bbox[3], reverse=True)[0]
                frame_bgr = swapper.get(frame_bgr, dst, src_face, paste_back=True)

        # HUD overlay
        if texture_mode != "skeleton":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img  = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img)
            draw.rectangle([0,0,W,22], fill=(0,0,0,200))
            draw.text((8,4), f"■ REC  CAM-04  {i/fps:05.2f}s", fill=(255,50,50))
            draw.text((W-170,4), "AI SURVEILLANCE ACTIVE", fill=(255,180,0))
            draw.rectangle([0,H-20,W,H], fill=(0,0,0,180))
            ts = datetime(2026,4,12,19,0,0) + timedelta(seconds=i/fps)
            draw.text((8,H-16), "CROWD INCIDENT DETECTED", fill=(80,255,100))
            draw.text((W-165,H-16), ts.strftime("%Y-%m-%d  %H:%M:%S"), fill=(180,180,180))
            frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        cv2.imwrite(f"{out_dir}/frame_{i:04d}.png", frame_bgr)

    renderer.delete()

    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", f"{out_dir}/frame_%04d.png",
        "-vf", f"scale={W}:{H}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        out_video
    ], check=True, capture_output=True)

    for f in os.listdir(out_dir): os.remove(os.path.join(out_dir, f))
    os.rmdir(out_dir)
