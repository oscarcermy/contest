import cv2
import os, json, glob
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import ColorVisuals
from collections import Counter
import itertools


# ========= 配置 =========
# 色彩 分级
PALETTE_23 = np.array([
    [128,  64, 128],   # 0  暗紫
    [232,  35, 244],   # 1  品红
    [ 70,  70,  70],   # 2  深灰
    [156, 102, 102],   # 3  玫瑰棕
    [153, 153, 190],   # 4  淡紫
    [  0, 220, 220],   # 5  青色
    [ 35, 142, 107],   # 6  蓝绿
    [142,   0,   0],   # 7  暗红
    [ 70,   0,   0],   # 8  更深红
    [  0,   0, 142],   # 9  纯蓝
    [  0,   0,  70],   # 10 深蓝
    [  0,  60, 100],   # 11 藏青
    [  0,  80, 100],   # 12 青灰
    [  0,   0, 230],   # 13 亮蓝
    [119,  11,  32],   # 14 酒红
    [  0, 255,   0],   # 15 高亮绿
    [255,   0,   0],   # 16 高亮红
    [255, 255,   0],   # 17 高亮黄
    [  0, 255, 255],   # 18 高亮青
    [255,   0, 255],   # 19 高亮品红
    [255, 128,   0],   # 20 橙色
    [128, 255, 128],   # 21 浅绿
    [255, 203, 219],   # 22 淡粉（新增）
    [0, 0, 0],   # 23 黑色（新增）
    [255, 255, 255]
], dtype=np.uint8)

OBJ_PATH     = "./gen3/demo_textured.glb"       # 你的原始网格（.obj / .ply / .glb 都可以）
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

def calculate_face_mask(input_face, mesh, cams_list, masks, classes):
    """
    计算每个面与每个相机成像平面的2D坐标的掩码类型。
    """
    face_masks = {}
    face_vertices = mesh.vertices[mesh.faces[input_face]]  # 获取面的顶点坐标
    
    mapping_count = []

    for idx, cam in enumerate([cams_list[0]]):
        mask, class_ = masks[idx], classes[idx]
        if mask.shape[1]==1280:
            mask = mask.transpose(0, 2, 1)
            mask = mask[:, :, ::-1]
        else:
            mask = mask[:, :, ::-1]
        #mask = np.transpose(mask, (0,2,1))  # (C,H,W) -> (C,W,H)
        #mask = mask[:, ::-1, :]

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
            
            world_to_cam = np.dot(R, vertex_homogeneous-t)  # 应用旋转，只取前三个元素
            world_to_cam = world_to_cam  # 应用平移
            cam_to_pixel = np.dot(K, world_to_cam)  # 转换到像素坐标
            pixel_coords = cam_to_pixel / cam_to_pixel[2]

            # 计算2D坐标
            u = int(round(pixel_coords[0]))
            v = int(round(pixel_coords[1]))

            # 检查投影是否在图像范围内
            if not (0 <= u < W and 0 <= v < H):
                continue

            scores = mask[:, v, u]
            if not any(scores):
                continue
            
            scores_idx = np.argmax(scores)
            mapping_count.append(class_[scores_idx])
    if mapping_count:
        mapping_count = Counter(mapping_count).most_common(1)[0][0]
    else:
        mapping_count = 23
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
    
    mask_files = [f"D:/learing____record/contest/preds/mask_"+"0"*(3-len(str(i)))+ str(i)+".npz" for i in range(36)]
    masks, classes = load_masks_from_npz(mask_files, (1280, 960))
    
    # 示例用法
    labels = []
    for i in range(40000):
        if i:
            input_face = i  # 要计算的面的索引
            labels.append(calculate_face_mask(input_face, mesh, cams_list, masks, classes))
            #labels.append(24)
        else:
            labels.append(23)

    # 1. 先强制顶点色（此时 visual.kind 一定是 'vertex'）
    rgb = PALETTE_23[labels]       
    rgba = np.column_stack([rgb, np.full((rgb.shape[0],1), 255, np.uint8)])
    mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=rgba)

    # 4. 导出
    mesh.export('semantic23.glb')


if __name__ == "__main__":
    main()
