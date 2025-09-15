import trimesh

# GLB 文件路径
GLB_PATH = "D:/learing____record/dataset/hy_gen/demo_textured.glb"

# 加载 GLB 文件
mesh = trimesh.load(GLB_PATH)

# 如果加载的是一个场景，提取所有网格对象
if isinstance(mesh, trimesh.Scene):
    total_faces = 0
    for geometry in mesh.geometry.values():
        total_faces += len(geometry.faces)
    print(f"Total number of faces in the GLB file: {total_faces}")
else:
    # 如果加载的是单个网格对象
    print(f"Number of faces in the mesh: {len(mesh.faces)}")