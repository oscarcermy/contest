import os
import math
import numpy as np
import trimesh
import pyrender
import imageio
from PIL import Image

# 文件路径
OBJ_PATH = "D:/learing____record/dataset/hy_gen/demo_textured.glb"  # 改成你的文件名
OUT_DIR = "renders"
os.makedirs(OUT_DIR, exist_ok=True)

# 纹理图片路径
ALBEDO_TEXTURE = "D:/learing____record/dataset/hy_gen/demo_textured.jpg"  # albedo 纹理
METALLIC_TEXTURE = "D:/learing____record/dataset/hy_gen/demo_textured_metallic.jpg"  # metallic 纹理
ROUGHNESS_TEXTURE = "D:/learing____record/dataset/hy_gen/demo_textured_roughness.jpg"  # roughness 纹理

# 1) 载入网格（尽量别自动重拓扑/重新计算UV）
mesh = trimesh.load(OBJ_PATH, force='mesh', process=False)

# 2) 加载纹理图片
if os.path.exists(ALBEDO_TEXTURE) and os.path.exists(METALLIC_TEXTURE) and os.path.exists(ROUGHNESS_TEXTURE):
    albedo_image = Image.open(ALBEDO_TEXTURE)
    metallic_image = Image.open(METALLIC_TEXTURE)
    roughness_image = Image.open(ROUGHNESS_TEXTURE)

    albedo_texture = np.array(albedo_image)  # 将 PIL.Image 转换为 numpy 数组
    metallic_texture = np.array(metallic_image)  # 将 PIL.Image 转换为 numpy 数组
    roughness_texture = np.array(roughness_image)  # 将 PIL.Image 转换为 numpy 数组

    # 创建材质并应用纹理
    material = pyrender.MetallicRoughnessMaterial(
        baseColorTexture=pyrender.Texture(
            sampler=pyrender.Sampler(),
            source=albedo_texture,
            source_channels='RGBA'  # 指定纹理通道
        ),
        metallicRoughnessTexture=pyrender.Texture(
            sampler=pyrender.Sampler(),
            source=np.dstack((metallic_texture, roughness_texture)),
            source_channels='RG'  # 指定纹理通道
        ),
        metallicFactor=1.0,  # 设置金属度为 1
        roughnessFactor=1.0  # 设置粗糙度为 1
    )
else:
    print("Texture images not found. Using default material.")
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.8, 0.8, 1.0],  # 灰色
        metallicFactor=0.0,  # 设置金属度为 0
        roughnessFactor=1.0  # 设置粗糙度为 1
    )

# 3) 构建场景（加一点环境光+方向光）
scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35, 0.35, 0.35])
pm = pyrender.Mesh.from_trimesh(mesh, material=material)  # 会带上 diffuse 贴图
scene.add(pm)
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light)

# 4) 相机与渲染器
W, H = 1280, 960
fx = fy = 1100.0
cx, cy = W / 2, H / 2
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
    theta = 2 * math.pi * i / NUM_VIEWS
    phi = math.radians(15)  # 15°俯视，避免完全水平造成遮挡严重
    cam_pos = center + np.array([
        radius * math.cos(theta) * math.cos(phi),
        radius * math.sin(phi),
        radius * math.sin(theta) * math.cos(phi),
    ])

    # 相机坐标系
    forward = (center - cam_pos); forward /= np.linalg.norm(forward)
    up = np.array([0, 1, 0], dtype=float)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # 世界->相机（R|t 使 X_cam = R*(X - C)）
    R_wc = np.stack([right, up, -forward], axis=0)
    t_wc = -R_wc @ cam_pos

    # 注意：pyrender节点姿态是相机在“世界中的位姿”，因此要用 [R^T | C]
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R_wc.T
    cam_pose[:3, 3] = cam_pos

    cam_node = scene.add(camera, pose=cam_pose)

    # 渲染彩色 + 深度
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    scene.remove_node(cam_node)

    # 保存
    imageio.imwrite(f"{OUT_DIR}/rgb_{i:03d}.png", color)
    np.save(f"{OUT_DIR}/depth_{i:03d}.npy", depth)

    cams_meta.append({
        "view_id": i,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "R": R_wc.tolist(),  # 世界->相机
        "t": t_wc.tolist(),  # 世界->相机
        "W": W, "H": H
    })

with open(f"{OUT_DIR}/cams.json", "w") as f:
    import json
    json.dump(cams_meta, f, indent=2)

print("Render done:", OUT_DIR)