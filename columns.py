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

# 相机参数
location = np.array([3.6, 2.07, 1.5])  # 相机位置
rotation = np.array([70, -0.0000001, 120])  # 相机旋转（度）
scale = np.array([1.0, 1.0, 1.0])  # 缩放

# 将旋转角度转换为弧度
rotation_rad = np.radians(rotation)

# 构建旋转矩阵
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(rotation_rad[0]), -np.sin(rotation_rad[0])],
    [0, np.sin(rotation_rad[0]), np.cos(rotation_rad[0])]
])

R_y = np.array([
    [np.cos(rotation_rad[1]), 0, np.sin(rotation_rad[1])],
    [0, 1, 0],
    [-np.sin(rotation_rad[1]), 0, np.cos(rotation_rad[1])]
])

R_z = np.array([
    [np.cos(rotation_rad[2]), -np.sin(rotation_rad[2]), 0],
    [np.sin(rotation_rad[2]), np.cos(rotation_rad[2]), 0],
    [0, 0, 1]
])

R = np.dot(R_z, np.dot(R_y, R_x))

# 相机内参
W, H = 1280, 960
fx = fy = 1100.0
cx, cy = W / 2, H / 2
camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

# 相机位姿
cam_pose = np.eye(4)
cam_pose[:3, :3] = R
cam_pose[:3, 3] = location

# 渲染器
renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

# 添加相机并渲染
scene.add(camera, pose=cam_pose)
color, depth = renderer.render(scene)

# 保存渲染结果
imageio.imwrite(f"{OUT_DIR}/columns_render.png", color)
np.save(f"{OUT_DIR}/depth.npy", depth)

print("Render done:", OUT_DIR)