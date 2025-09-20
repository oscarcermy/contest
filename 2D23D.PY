# backproject_and_fuse.py
import os, json, glob
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import ColorVisuals

# ========= 配置 =========
OBJ_PATH     = "D:/learing____record/dataset/hy_gen/demo_textured.glb"       # 你的原始网格（.obj / .ply / .glb 都可以）
REN_DIR      = "renders"       # 含：rgb_*.png / deep_*.npy / cams.json
PRED_DIR     = "preds"         # 含：mask_*.npz / preds.json
DEPTH_PREFIX = "depth"          # 深度文件前缀，如 deep_000.npy
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
def depth_consistency_score(V, cam, depth, sample_max=8000):
    """与这台相机的深度一致性：robust 线性拟合 z_img ≈ s*Z + b 后的残差中位数（越小越好）"""
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H = cam["W"], cam["H"]
    R = np.array(cam["R"]); t = np.array(cam["t"])

    # 你的多变体投影（已定义）
    u_i, v_i, Z, valid, pick, cnt = project_variants(V, R, t, fx, fy, cx, cy, W, H)
    if cnt == 0:
        return np.inf, 0, u_i, v_i, Z, valid, pick

    u_clip = np.clip(u_i, 0, W-1)
    v_clip = np.clip(v_i, 0, H-1)
    valid = valid & (u_i == u_clip) & (v_i == v_clip) & (Z > 0)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.inf, 0, u_i, v_i, Z, valid, pick

    z_img = depth[v_clip[idx], u_clip[idx]]

    # 有些渲染深度会有 0/NaN/Inf，剔除
    good = np.isfinite(z_img) & (z_img > 0)
    if good.sum() < 200:
        return np.inf, 0, u_i, v_i, Z, valid, pick

    Zi   = Z[idx][good]
    Di   = z_img[good]

    # 采样，避免超大数组回归慢
    if Zi.shape[0] > sample_max:
        sel = np.random.default_rng(0).choice(Zi.shape[0], sample_max, replace=False)
        Zi, Di = Zi[sel], Di[sel]

    # robust 一阶拟合：Di ≈ s*Zi + b
    # 用最小二乘先求 s,b，然后以中位绝对残差作为分数
    A = np.column_stack([Zi, np.ones_like(Zi)])
    s, b = np.linalg.lstsq(A, Di, rcond=None)[0]
    resid = np.abs(s*Zi + b - Di)
    score = float(np.median(resid))  # 尺度无关

    return score, int(idx.size), u_i, v_i, Z, valid, pick
def best_cam_by_depth(V, cams_list, depth):
    """在所有相机里选残差最小的；若都不理想，返回 None 让主流程走 fallback。"""
    best = {"score": np.inf, "cnt": 0}
    for cam in cams_list:
        score, cnt, u_i, v_i, Z, valid, pick = depth_consistency_score(V, cam, depth)
        if (score < best["score"]) or (np.isinf(best["score"]) and np.isinf(score) and cnt > best["cnt"]):
            best.update(dict(cam=cam, u=u_i, v=v_i, Z=Z, valid=valid, pick=pick, cnt=cnt, score=score))
    return None if ("cam" not in best) else best


def resize_masks_to(masks, H, W):
    """将 [N,h,w] 的掩码最近邻放大到 [N,H,W]；空则原样返回"""
    if masks.size == 0:
        return masks
    h, w = masks.shape[-2], masks.shape[-1]
    if (h, w) == (H, W):
        return (masks > 0.5)
    up = []
    for inst in masks:
        im = Image.fromarray((inst > 0.5).astype(np.uint8) * 255)
        im = im.resize((W, H), resample=Image.NEAREST)
        up.append(np.array(im) > 127)
    return np.stack(up, axis=0)

def maybe_denorm_intrinsics(fx, fy, cx, cy, W, H):
    # 若 cx,cy 在 0~2 范围内，视为归一化主点，乘 W,H
    if abs(cx) <= 2 and abs(cy) <= 2:
        cx, cy = cx * W, cy * H
    # 若 fx,fy 很小，视为归一化焦距，乘 W,H
    if fx < 10: fx *= W
    if fy < 10: fy *= H
    return fx, fy, cx, cy

def project_variants(V, R, t, fx, fy, cx, cy, W, H):
    """尝试多外参/内参变体，返回 valid 数最多的结果。"""
    Vt = V.T
    intri_cands = [
        (fx, fy, cx, cy),
        maybe_denorm_intrinsics(fx, fy, cx, cy, W, H),
    ]
    best = dict(cnt=-1)

    for idx_intri, (Fx, Fy, Cx, Cy) in enumerate(intri_cands):
        # A: Xc = R X + t  （R,t=world->cam）
        Xc = R @ Vt + t.reshape(3,1)
        ZA = Xc[2]
        u = Fx*(Xc[0]/ZA) + Cx; v = Fy*(Xc[1]/ZA) + Cy
        uAi = np.round(u).astype(int); vAi = np.round(v).astype(int)
        validA = (ZA>0)&(uAi>=0)&(uAi<W)&(vAi>=0)&(vAi<H); cntA = int(validA.sum())
        if cntA > best.get('cnt', -1):
            best = dict(u=uAi, v=vAi, Z=ZA, valid=validA, cnt=cntA, pick=f"A/intri{idx_intri}")

        # B: Xc = R^T (X - C)（R=cam->world 旋转，t=C 相机中心）
        Xc = R.T @ (Vt - t.reshape(3,1))
        ZB = Xc[2]
        u = Fx*(Xc[0]/ZB) + Cx; v = Fy*(Xc[1]/ZB) + Cy
        uBi = np.round(u).astype(int); vBi = np.round(v).astype(int)
        validB = (ZB>0)&(uBi>=0)&(uBi<W)&(vBi>=0)&(vBi<H); cntB = int(validB.sum())
        if cntB > best.get('cnt', -1):
            best = dict(u=uBi, v=vBi, Z=ZB, valid=validB, cnt=cntB, pick=f"B/intri{idx_intri}")

        # C: Xc = R (X - C)（R=world->cam 旋转，t=C 相机中心）
        Xc = R @ (Vt - t.reshape(3,1))
        ZC = Xc[2]
        u = Fx*(Xc[0]/ZC) + Cx; v = Fy*(Xc[1]/ZC) + Cy
        uCi = np.round(u).astype(int); vCi = np.round(v).astype(int)
        validC = (ZC>0)&(uCi>=0)&(uCi<W)&(vCi>=0)&(vCi<H); cntC = int(validC.sum())
        if cntC > best.get('cnt', -1):
            best = dict(u=uCi, v=vCi, Z=ZC, valid=validC, cnt=cntC, pick=f"C/intri{idx_intri}")

        # D: Xc = R^T X + t（R=cam->world 旋转，t=已是 world->cam 平移）
        Xc = R.T @ Vt + t.reshape(3,1)
        ZD = Xc[2]
        u = Fx*(Xc[0]/ZD) + Cx; v = Fy*(Xc[1]/ZD) + Cy
        uDi = np.round(u).astype(int); vDi = np.round(v).astype(int)
        validD = (ZD>0)&(uDi>=0)&(uDi<W)&(vDi>=0)&(vDi<H); cntD = int(validD.sum())
        if cntD > best.get('cnt', -1):
            best = dict(u=uDi, v=vDi, Z=ZD, valid=validD, cnt=cntD, pick=f"D/intri{idx_intri}")

    return best['u'], best['v'], best['Z'], best['valid'], best['pick'], best['cnt']

def best_cam_for_view(V, cams_list):
    """当 view_id 与图像可能错位时：在所有相机中选择 valid 最大的那一台。"""
    best = None
    for cam in cams_list:
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        W, H = cam["W"], cam["H"]
        R = np.array(cam["R"]); t = np.array(cam["t"])
        u_i, v_i, Z, valid, pick, cnt = project_variants(V, R, t, fx, fy, cx, cy, W, H)
        if best is None or cnt > best["cnt"]:
            best = dict(cam=cam, u=u_i, v=v_i, Z=Z, valid=valid, pick=pick, cnt=cnt, W=W, H=H)
    return best

def main():
    # 1) 读网格
    mesh = load_mesh(OBJ_PATH)
    V = mesh.vertices.view(np.ndarray)  # (Nv,3)
    """是从加载的3D网格（mesh）中提取的顶点（vertex）数组。这里的 mesh.vertices 是一个包含网格中所有顶点的Trimesh对象属性，每个顶点由其在三维空间中的坐标（x, y, z）表示。
    当我们调用 mesh.vertices.view(np.ndarray)，我们实际上是在请求Trimesh库将顶点数据转换为一个NumPy数组，这样我们就可以利用NumPy提供的高效数值计算功能。数组 V 的形状是 (Nv, 3)，其中 Nv 是网格中顶点的数量，而 3 表示每个顶点的三维坐标（x, y, z）。"""

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

    votes = np.zeros((V.shape[0], K), dtype=np.float32)

    mask_files = sorted(glob.glob(os.path.join(PRED_DIR, "mask_*.npz")))
    print(f"[INFO] Found {len(mask_files)} mask files")

    total_vis_hits = 0

    for mf in mask_files:
        vid = int(os.path.basename(mf).split("_")[1].split(".")[0])

        # 选相机：正常情况下按 view_id；若错位则启用最佳匹配
        vid = int(os.path.basename(mf).split("_")[1].split(".")[0])

# 先读这张图自己的深度
        depth_path = os.path.join(REN_DIR, f"{DEPTH_PREFIX}_{vid:03d}.npy")
        if not os.path.exists(depth_path):
            print(f"[WARN] missing depth: {depth_path}, skip")
            continue
        depth = np.load(depth_path)

        # 用这张 depth 在所有相机里挑“最像”的那一台
        best = best_cam_by_depth(V, list(cams_map.values()), depth)
        if best is None or best["cnt"] < 200 or np.isinf(best["score"]):
            # 回退：用“valid 最大”的相机（你之前能跑通的办法）
            fallback = best_cam_for_view(V, list(cams_map.values()))
            if fallback is None or fallback["cnt"] == 0:
                print(f"[VIEW {vid:03d}] no camera matches depth, and fallback invalid, skip")
                continue
            cam = fallback["cam"]; W, H = fallback["W"], fallback["H"]
            u_i, v_i, Z, valid = fallback["u"], fallback["v"], fallback["Z"], fallback["valid"]
            print(f"[VIEW {vid:03d}] FALLBACK USE cam_id={cam['view_id']} pick={fallback['pick']} valid={fallback['cnt']}")
        else:
            cam = best["cam"]; W, H = cam["W"], cam["H"]
            u_i, v_i, Z, valid = best["u"], best["v"], best["Z"], best["valid"]
            print(f"[VIEW {vid:03d}] USE cam_id={cam['view_id']} pick={best['pick']} valid={best['cnt']} depth-score={best['score']:.4g}")


        # 读深度
        depth_path = os.path.join(REN_DIR, f"{DEPTH_PREFIX}_{vid:03d}.npy")
        if not os.path.exists(depth_path):
            # 若启用最佳相机，但深度文件仍以原 vid 命名，此处仍用原 vid 的深度（与 rgb 对齐）
            if not os.path.exists(depth_path):
                print(f"[WARN] missing depth: {depth_path}, skip view")
                continue
        depth = np.load(depth_path)
        if depth.shape != (H, W):
            print(f"[WARN] depth shape {depth.shape} != ({H},{W}), will clip indices later.")

        # 读分割
        data = np.load(mf, allow_pickle=True)
        masks = data["masks"]                   # [N,h,w] 或空
        classes = data["classes"]               # [N]
        confs = data["confs"] if "confs" in data.files else None  # [N]（可能不存在）

        # 掩码对齐尺寸
        masks = resize_masks_to(masks, H, W)

        # 可见性（Z vs depth）
        u_clip = np.clip(u_i, 0, W-1)
        v_clip = np.clip(v_i, 0, H-1)
        valid = valid & (u_i == u_clip) & (v_i == v_clip) & (Z > 0)
        idx = np.where(valid)[0]
        if idx.size == 0:
            print(f"[VIEW {vid:03d}] no valid pixels after clip")
            continue

        # 取此视角对应像素的深度
        z_img = depth[v_clip[idx], u_clip[idx]]

    # 1) 忽略无效深度：0、NaN、Inf
        good = np.isfinite(z_img) & (z_img > 1e-6)

    # 2) 相对+绝对阈值（建议：ABS_EPS=0.05~0.10, REL_EPS=0.10~0.20）
        abs_eps = 0.08   # ≈ 8 cm
        rel_eps = 0.15   # 允许 15% 的相对误差

        vis = np.zeros_like(valid)
        if good.any():
            err = np.abs(Z[idx][good] - z_img[good])
            thr = np.maximum(abs_eps, rel_eps * np.abs(z_img[good]))
            ok  = err < thr
        # 只在“深度有效且误差可接受”的像素处判可见
            vis_idx = idx[good][ok]
            vis[vis_idx] = True

            vis_cnt = int(vis.sum())

        # 计票
        if masks.size != 0 and classes.size != 0:
            for k, (inst_mask, cls_id) in enumerate(zip(masks, classes)):
                key = str(int(cls_id)) if str(int(cls_id)) in names else int(cls_id)
                cls_name = names[key]
                cid = name2id.get(cls_name, 0)
                if cid == 0: 
                    continue
                sel = vis
                if sel.any():
                    hit = np.zeros(V.shape[0], dtype=bool)
                    hit[sel] = inst_mask[v_clip[sel], u_clip[sel]]
                    weight = float(confs[k]) if (confs is not None and k < len(confs)) else 1.0
                    votes[hit, cid] += weight
                    total_vis_hits += int(hit.sum())
        print(f"[VIEW {vid:03d}] vis={vis_cnt}, masks={0 if masks.size==0 else masks.shape[0]}")

    # 最终标签
    labels = votes.argmax(axis=1)

    # 上色并导出

    # labels: (Nv,) 你的最终类别ID
    # name2id: 类别到ID的映射
    rng = np.random.default_rng(SEED)
    palette = np.vstack([[160,160,160], rng.integers(0,255,size=(len(name2id),3))]).astype(np.uint8)

    rgb = palette[labels]                                  # (Nv,3)
    rgba = np.column_stack([rgb, np.full((rgb.shape[0],1), 255, np.uint8)])  # (Nv,4)

    # 关键：强制用 ColorVisuals（而不是 TextureVisuals）
    mesh.visual = ColorVisuals(mesh, vertex_colors=rgba)
    print(mesh.visual.kind) 
    # 导出 GLB（带 COLOR_0 顶点属性）
    mesh.export(OUT_GLB)

if __name__ == "__main__":
    main()



def parse_yolo_output(yolo_output_path, image_shape):
    """
    解析YOLO输出文件，返回检测到的对象的类别、置信度和边界框坐标。
    """
    # 假设YOLO输出格式为：<class_id> <confidence> <x_center> <y_center> <width> <height>
    

    """ detections = []
    for line in lines:
        class_id, confidence, x_center, y_center, width, height = map(float, line.strip().split())
        x1 = int((x_center - width / 2) * image_shape[1])
        y1 = int((y_center - height / 2) * image_shape[0])
        x2 = int((x_center + width / 2) * image_shape[1])
        y2 = int((y_center + height / 2) * image_shape[0])
        detections.append((class_id, confidence, (x1, y1, x2, y2))) """

    return 1

def apply_yolo_masks_ori(mask_files, image_shape):
    """
    应用YOLO输出的分割掩码。
    """
    masks = []
    for mf in mask_files:
        vid = int(os.path.basename(mf).split("_")[1].split(".")[0])
        yolo_output_path = os.path.join(PRED_DIR, f"mask_{vid:03d}.npz")  # 假设YOLO输出文件名格式
        detections = parse_yolo_output(yolo_output_path, image_shape)
        
    """     # 创建一个全黑的掩码图像
        mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

        for class_id, confidence, (x1, y1, x2, y2) in detections:
            # 根据检测到的对象绘制掩码
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # 255表示白色，-1表示填充

        masks.append(mask)

    return masks """