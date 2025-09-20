import os
import numpy as np
import trimesh
import pyrender
import imageio

# 文件路径
OBJ_PATH = "semantic23.glb"  # 改成你的文件路径
OUT_DIR = "renders"
os.makedirs(OUT_DIR, exist_ok=True)

# 载入网格
mesh = trimesh.load(OBJ_PATH, force='mesh', process=False)

# 创建材质
material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.8, 0.8, 0.8, 1.0],
    metallicFactor=0.0,
    roughnessFactor=1.0
)

# 构建场景
scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35, 0.35, 0.35])
pm = pyrender.Mesh.from_trimesh(mesh, material=material)
scene.add(pm)
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light)

# 相机参数
fx = 1100.0
fy = 1100.0
cx = 640.0
cy = 480.0
R = np.array([
    [-0.0, 0.0, -1.0],
    [-0.2588190451025208, 0.9659258262890684, 0.0],
    [0.9659258262890684, 0.2588190451025208, -0.0]
])
t = np.array([ 2.304415, 0.61254757, -0.00247851])
W = 1280
H = 960

# 创建相机
camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

# 相机位姿
cam_pose = np.eye(4)
cam_pose[:3, :3] = R.T
cam_pose[:3, 3] = t

# 渲染器
renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

# 添加相机并渲染
scene.add(camera, pose=cam_pose)
color, depth = renderer.render(scene)

# 保存渲染结果
imageio.imwrite(f"{OUT_DIR}/render.png", color)
np.save(f"{OUT_DIR}/depth.npy", depth)

print("Render done:", OUT_DIR)

# 可视化相机方向
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制坐标轴
ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=1, arrow_length_ratio=0.1)  # X轴
ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=1, arrow_length_ratio=0.1)  # Y轴
ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=1, arrow_length_ratio=0.1)  # Z轴

# 计算相机方向向量
camera_direction = R @ np.array([0, 0, 1])
camera_position = t

# 绘制相机方向
ax.quiver(camera_position[0], camera_position[1], camera_position[2],
          camera_direction[0], camera_direction[1], camera_direction[2],
          color='k', length=1, arrow_length_ratio=0.1)

# 设置图例
ax.legend(['X', 'Y', 'Z', 'Camera Direction'])

# 设置坐标轴范围
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])

# 显示图形
plt.show()