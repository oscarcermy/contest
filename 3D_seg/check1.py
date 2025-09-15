import numpy as np, trimesh

m = trimesh.load("car_semantic.glb", force='mesh')
print("is Trimesh:", isinstance(m, trimesh.Trimesh))
print("vertex_colors attr:", hasattr(m.visual, "vertex_colors"))
if hasattr(m.visual, "vertex_colors"):
    print("vertex_colors shape:", m.visual.vertex_colors.shape)  # (Nv, 4)
    uniq = np.unique(m.visual.vertex_colors[:,:3], axis=0)
    print("unique colors:", len(uniq))
