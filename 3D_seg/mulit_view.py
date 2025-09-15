# render_with_textures.py
import os, json, math, numpy as np, trimesh, pyrender, imageio

OBJ_PATH = "D:/learing____record/dataset/hy_gen/white_mesh_remesh.obj"  # 改成你的文件名
OUT_DIR = "renders"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) 载入网格（尽量别自动重拓扑/重新计算UV）
mesh = trimesh.load(OBJ_PATH, force='mesh', process=False)

# --- 如果发现渲染出来是纯灰色：说明 MTL 没把 albedo 带进来，手动指定一张颜色贴图 ---
ALBEDO_FALLBACK = "sample/texture_pbr_v128.png"   # 你的彩色贴图（疑似 albedo）
if hasattr(mesh.visual, "material") and getattr(mesh.visual.material, "image", None) is None:
    if os.path.exists(ALBEDO_FALLBACK):
        img = imageio.imread(ALBEDO_FALLBACK)
        mesh.visual.material.image = img  # 手动挂 albedo
# -----------------------------------------------------------------------

# 2) 构建场景（加一点环境光+方向光）
scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.35,0.35,0.35])
pm = pyrender.Mesh.from_trimesh(mesh, smooth=True)  # 会带上 diffuse 贴图
scene.add(pm)
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light)

# 3) 相机与渲染器
W, H = 1280, 960
fx = fy = 1100.0
cx, cy = W/2, H/2
camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

# 环绕相机半径
bbox = mesh.bounding_box.extents
radius = float(np.linalg.norm(bbox)) * 0.9
center = mesh.bounding_box.centroid

cams_meta = []
NUM_VIEWS = 36
for i in range(NUM_VIEWS):
    # 采样方位角 + 轻微俯仰
    theta = 2*math.pi * i/NUM_VIEWS
    phi = math.radians(15)   # 15°俯视，避免完全水平造成遮挡严重
    cam_pos = center + np.array([
        radius*math.cos(theta)*math.cos(phi),
        radius*math.sin(phi),
        radius*math.sin(theta)*math.cos(phi),
    ])

    # 相机坐标系
    forward = (center - cam_pos); forward /= np.linalg.norm(forward)
    up = np.array([0,1,0], dtype=float)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # 世界->相机（R|t 使 X_cam = R*(X - C)）
    R_wc = np.stack([right, up, -forward], axis=0)
    t_wc = -R_wc @ cam_pos

    # 注意：pyrender节点姿态是相机在“世界中的位姿”，因此要用 [R^T | C]
    cam_pose = np.eye(4)
    cam_pose[:3,:3] = R_wc.T
    cam_pose[:3, 3] = cam_pos

    cam_node = scene.add(camera, pose=cam_pose)

    # 渲染彩色 + 深度
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    # 单独取深度：也可以直接用上面的 depth
    scene.remove_node(cam_node)

    # 保存
    imageio.imwrite(f"{OUT_DIR}/rgb_{i:03d}.png", color)
    np.save(f"{OUT_DIR}/depth_{i:03d}.npy", depth)

    cams_meta.append({
        "view_id": i,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "R": R_wc.tolist(),       # 世界->相机
        "t": t_wc.tolist(),       # 世界->相机
        "W": W, "H": H
    })

with open(f"{OUT_DIR}/cams.json", "w") as f:
    json.dump(cams_meta, f, indent=2)

print("Render done:", OUT_DIR)
