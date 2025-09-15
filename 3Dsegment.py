import cv2
import os, json, glob
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import ColorVisuals
from collections import Counter
import itertools

# ========= 配置 =========
OBJ_PATH     = "D:/learing____record/dataset/hy_gen/demo_textured.glb"       # 你的原始网格（.obj / .ply / .glb 都可以）
REN_DIR      = "./renders"       # 含：rgb_*.png / deep_*.npy / cams.json
PRED_DIR     = "./preds"         # 含：mask_*.npz / preds.json
DEPTH_PREFIX = "./depth"          # 深度文件前缀，如 deep_000.npy
OUT_GLB      = "car_semantic.glb"
USE_BEST_CAM = False           # 若 view_id 与图像编号有错位，可设 True 自动匹配最佳相机
ABS_EPS      = 0.12            # 可见性：绝对阈值（按你的模型尺度≈厘米可再调）
REL_EPS      = 0.2            # 可见性：相对阈值（占深度的百分比）
SEED         = 42              # 调色板随机种子（固定颜色）
# =======================

def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # 可能是场景，取第一几何
        if hasattr(mesh, 'geometry') and len(mesh.geometry):
            mesh = list(mesh.geometry.values())[0]
        else:
            raise ValueError("Failed to load a mesh from OBJ/GLB.")
    return mesh

def calculate_face_mask(input_face, mesh, cams_list):
    """
    计算每个面与每个相机成像平面的2D坐标的掩码类型。
    """
    face_masks = {}
    face_vertices = mesh.vertices[mesh.faces[input_face]]  # 获取面的顶点坐标
    
    mask_files = [f"D:/learing____record/contest/preds/mask_"+"0"*(3-len(str(i)))+ str(i)+".npz" for i in range(36)]

    mapping_count = []
    masks, classes = load_masks_from_npz(mask_files, (1280, 960))
    for idx, cam in enumerate(cams_list):
        mask, class_ = masks[idx], classes[idx]

        # 提取相机参数
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        W, H = cam["W"], cam["H"]
        R = np.array(cam["R"])  # 3x3 旋转矩阵
        t = np.array(cam["t"])  # 平移向量

        # 构建内参矩阵
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        
        # 将面的顶点投影到成像平面
        for vertex in face_vertices:
            vertex_homogeneous = vertex  # 转换为齐次坐标
            world_to_cam = np.dot(R, vertex_homogeneous)  # 应用旋转，只取前三个元素
            world_to_cam = world_to_cam + t  # 应用平移
            cam_to_pixel = np.dot(K, world_to_cam)  # 转换到像素坐标
            pixel_coords = cam_to_pixel / cam_to_pixel[2]

            # 计算2D坐标
            u = int(round(pixel_coords[0]))
            v = int(round(pixel_coords[1]))

            # 检查投影是否在图像范围内
            if not (0 <= u < H and 0 <= v < W):
                continue

            scores = mask[:, u, v]
            if not any(scores):
                continue

            scores_idx = np.argmax(mask[:, u, v])
            mapping_count.append(class_[scores_idx])

    mapping_count = Counter(mapping_count).most_common(1)[0][0]
            
    return mapping_count

def load_masks_from_npz(npz_files, image_shapes):
    """
    从npz文件中加载分割掩码。

    参数:
    npz_files -- npz文件路径列表
    image_shapes -- 对应图像的尺寸列表 [(height, width), ...]

    返回:
    masks -- 分割掩码列表
    """
    masks = []
    classes = []
    for file in npz_files:
        data = np.load(file, allow_pickle=True)
        masks.append(data['masks'])
        classes.append(data['classes'])
    return masks, classes 

def main():
    # 1) 读网格
    mesh = load_mesh(OBJ_PATH)
    V = mesh.vertices.view(np.ndarray)  # (Nv,3)

    # 2) 相机参数
    with open(os.path.join(REN_DIR, "cams.json"), "r", encoding="utf-8") as f:
        cams = json.load(f)
    cams_map = {c["view_id"]: c for c in cams}
    cams_list = list(cams_map.values())

    # 3) 类别映射
    with open(os.path.join(PRED_DIR, "preds.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    names = meta["names"]  # 可能是 {"0":"wheel",...}
    id_sorted = sorted(map(int, names.keys()))
    all_names = [names[str(k)] for k in id_sorted]
    name2id = {n: i+1 for i, n in enumerate(all_names)}  # 0 留作“未标注”
    K = len(name2id) + 1

    import time
    start = time.time()
    # 示例用法
    for i in range(20):
        input_face = i  # 要计算的面的索引
        face_masks = calculate_face_mask(input_face, mesh, cams_list)
    print(time.time()-start)

if __name__ == "__main__":
    main()
