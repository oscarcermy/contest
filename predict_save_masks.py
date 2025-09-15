# predict_save_masks.py
from ultralytics import YOLO
import numpy as np
import glob, os, json

# ====== 配置区 ======
WEIGHTS = "./best.pt"  # 你的微调权重
IN_DIR  = "renders"                                # 输入渲染图目录
OUT_DIR = "preds"                                  # 输出 npz 和元信息目录
IMGSZ   = 1280                                     # 推理分辨率(小部件建议>=1024)
CONF    = 0.3                                      # 置信度阈值
IOU     = 0.5                                      # NMS IOU 阈值
DEVICE  = "0"                                      # "0" 用GPU; 无GPU用 "cpu"
SAVE_VIS = False                                    # 是否保存可视化到 runs/segment/predict*
# ====================

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(WEIGHTS)
names = {int(k): v for k, v in model.model.names.items()}

rgb_paths = sorted(glob.glob(os.path.join(IN_DIR, "rgb_*.png")))
meta_views = []

print(f"[INFO] Found {len(rgb_paths)} images in {IN_DIR}")
for p in rgb_paths:
    # 解析视角ID（rgb_003.png -> 003）
    base = os.path.basename(p)
    vid = int(base.split("_")[1].split(".")[0])

    # 预测
    results = model(
        source=p,
        device=DEVICE, save=SAVE_VIS, retina_masks=True, verbose=False
    )
    r = results[0]

    # 取掩模与类别ID
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()        # [N, H, W]，实例掩模
    else:
        masks = np.zeros((0,), dtype=np.uint8)

    if r.boxes is not None and r.boxes.cls is not None:
        classes = r.boxes.cls.cpu().numpy().astype(int)  # [N]，每个实例的类别索引
    else:
        classes = np.array([], dtype=int)

    # 保存供回投脚本使用
    np.savez_compressed(
        os.path.join(OUT_DIR, f"mask_{vid:03d}.npz"),
        masks=masks,
        classes=classes
    )

    # 记录这一视角检测到的类别名（可用于检查）
    meta_views.append({
        "view_id": vid,
        "classes": [names[int(c)] for c in classes]
    })

# 保存类别映射与视角汇总
with open(os.path.join(OUT_DIR, "preds.json"), "w", encoding="utf-8") as f:
    json.dump({"names": names, "views": meta_views}, f, ensure_ascii=False, indent=2)

print(f"[DONE] Saved masks to: {OUT_DIR}")
print(f"[TIP ] Visualizations: runs/segment/predict*/ (若 SAVE_VIS=True)")
