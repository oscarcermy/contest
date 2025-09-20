import pyrender
import trimesh
import numpy as np

# 1. 加载带顶点色的网格
mesh = trimesh.load('semantic23.glb', force='mesh')
print(mesh.visual.kind)                 # → vertex
print(mesh.visual.vertex_colors.shape)  # → (Nv, 4)
#assert mesh.visual.kind == 'vertex'  # 确保有顶点色

# 2. 转成 pyrender.Mesh
pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

# 3. 场景 + 相机 + 光源
scene = pyrender.Scene()
scene.add(pr_mesh)

# 简单相机（z=3 米，看向原点）
cam = pyrender.PerspectiveCamera(yfov=np.pi/3)
cam_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 3],
    [0, 0, 0, 1]
])
scene.add(cam, pose=cam_pose)

# 白光
light = pyrender.DirectionalLight(color=[1,1,1], intensity=3.0)
scene.add(light, pose=cam_pose)

# 4. 渲染
r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
color, _ = r.render(scene)
r.delete()

# 5. 保存或显示
import PIL.Image as Image
Image.fromarray(color).save('./semantic23_pyrender.png')

print('Done → semantic23_pyrender.png')