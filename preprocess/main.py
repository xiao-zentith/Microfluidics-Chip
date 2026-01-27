# main.py
import os
import cv2
import numpy as np
import config
from detector import ChamberDetector
from utils import CrossGeometryEngine

def find_chip_id_from_path(file_path):
    """从路径中提取 chip ID，例如 'chip1'"""
    parts = file_path.split(os.sep)
    for part in parts:
        if part.startswith("chip"): 
            return part
    return None

def main():
    # 1. 初始化
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DEBUG_DIR, exist_ok=True)
    
    detector = ChamberDetector()      # 加载 YOLO
    engine = CrossGeometryEngine()    # 加载新版几何引擎
    
    gt_cache = {} # 缓存 GT 切片 {chip_id: slices}

    print(f"Scanning raw images in: {config.INPUT_RAW_ROOT}")

    # 2. 遍历文件
    for root, dirs, files in os.walk(config.INPUT_RAW_ROOT):
        for fname in files:
            if not (fname.lower().endswith((".jpg", ".png", ".jpeg"))):
                continue
            
            raw_path = os.path.join(root, fname)
            chip_id = find_chip_id_from_path(raw_path)
            
            if chip_id is None: 
                print(f"[Skip] No chip ID found in path: {raw_path}")
                continue

            # --- 处理 GT (Ground Truth) ---
            if chip_id not in gt_cache:
                gt_filename = f"{chip_id}.jpg" # 假设 GT 命名规则
                gt_path = os.path.join(config.GT_LIBRARY_DIR, gt_filename)
                
                # 尝试找 png
                if not os.path.exists(gt_path):
                    gt_path = gt_path.replace(".jpg", ".png")
                
                if os.path.exists(gt_path):
                    print(f"--- Processing GT for {chip_id} ---")
                    img_gt = cv2.imread(gt_path)
                    # 1. Detect
                    pts_gt, cls_gt = detector.detect(img_gt)
                    # 2. Process (Align & Slice)
                    slices_gt, vis_gt = engine.process(img_gt, pts_gt, cls_gt)
                    
                    if slices_gt is not None:
                        gt_cache[chip_id] = slices_gt
                        # 保存 GT 的 Debug 图看一看是不是也正了
                        cv2.imwrite(os.path.join(config.DEBUG_DIR, f"GT_{chip_id}.jpg"), vis_gt)
                    else:
                        print(f"[Error] GT Alignment Failed: {chip_id}")
                else:
                    print(f"[Warn] GT file missing: {gt_path}")

            # --- 处理 Raw (Non-Standard) ---
            if chip_id in gt_cache: # 只有找到了 GT 才能处理配对
                print(f"Processing Raw: [{chip_id}] {fname}")
                img_raw = cv2.imread(raw_path)
                
                # 1. Detect
                pts_raw, cls_raw = detector.detect(img_raw)
                
                # 2. Process (Align & Slice)
                slices_raw, vis_raw = engine.process(img_raw, pts_raw, cls_raw)
                
                if slices_raw is not None:
                    # 保存可视化图
                    unique_name = f"{chip_id}_{fname}"
                    cv2.imwrite(os.path.join(config.DEBUG_DIR, f"vis_{unique_name}"), vis_raw)
                    
                    # 保存配对数据 (.npy)
                    save_name = unique_name.replace(".jpg", ".npy").replace(".png", ".npy")
                    save_path = os.path.join(config.OUTPUT_DIR, save_name)
                    
                    np.save(save_path, {
                        'raw_slices': slices_raw, # (12, 64, 64, 3)
                        'gt_slices': gt_cache[chip_id], # (12, 64, 64, 3)
                        'chip_id': chip_id
                    })

                    # --- 验证代码 ---
                    # 创建一个临时文件夹看切片顺序
                    debug_slice_dir = os.path.join(config.DEBUG_DIR, "slice_check", unique_name)
                    os.makedirs(debug_slice_dir, exist_ok=True)

                    for i in range(12):
                        # 保存 Raw 切片 (比如 0_raw.jpg)
                        cv2.imwrite(os.path.join(debug_slice_dir, f"{i}_raw.jpg"), slices_raw[i])
                        # 保存 GT 切片 (比如 0_gt.jpg)
                        cv2.imwrite(os.path.join(debug_slice_dir, f"{i}_gt.jpg"), gt_cache[chip_id][i])

                    print(f"-> Saved: {save_name}")
                else:
                    print(f"-> Failed: Alignment Error")
            else:
                pass # 没有 GT 跳过

if __name__ == "__main__":
    main()