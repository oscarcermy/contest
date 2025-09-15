import pyrender
import trimesh
import numpy as np
from PIL import Image
import os

# 加载 GLB 文件
scene = trimesh.load('D:/learing____record/dataset/hy_gen/demo_textured.glb')



# 如果加载的是一个场景，提取第一个网格对象
if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=m.vertices, faces=m.faces, visual=m.visual) for m in scene.geometry.values()])
else:
    mesh = scene

# 加载纹理图片
texture_image = Image.open("D:/learing____record/dataset/hy_gen/demo_textured.jpg")


texture_image = np.array(texture_image)  # 将 PIL.Image 转换为 numpy 数组

# 创建材质并应用纹理
material = pyrender.MetallicRoughnessMaterial(
    baseColorTexture=pyrender.Texture(
        sampler=pyrender.Sampler(),
        source=texture_image,
        source_channels='RGBA'  # 指定纹理通道
    ),
    metallicFactor=0.5,
    roughnessFactor=0.5
)

# 将材质应用到模型
mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material)

# 创建场景
scene = pyrender.Scene()

# 添加模型到场景
scene.add(mesh_node)

# 创建相机
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -5.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
scene.add(camera, pose=camera_pose)

# 创建灯光
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.0)
scene.add(light, pose=camera_pose)

# 渲染
r = pyrender.OffscreenRenderer(640, 480)
color, depth = r.render(scene)

# 保存渲染结果
output_path = 'rendered_image.png'
Image.fromarray(color).save(output_path)
print(f"Rendered image saved to {output_path}")