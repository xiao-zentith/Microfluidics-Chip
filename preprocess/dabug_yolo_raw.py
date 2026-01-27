# debug_yolo_raw.py

import cv2
import numpy as np
import os
import config
from detector import ChamberDetector
from utils import CrossGeometryEngine # 引入我们最新的 V10.0 引擎

def main():
    # --- 1. 设置要测试的图片路径 ---
    # 请在这里替换为你想要 Debug 的具体图片路径 (Raw 图或 GT 图均可)
    img_path = "/home/asus515/PycharmProjects/YOLO_v11/dataset/preprocess/raw_data/chip3/S003_dirty_09_pred.jpg" 
    
    # 检查路径是否存在
    if not os.path.exists(img_path):
        print(f"[Error] 找不到图片: {img_path}")
        print("请修改代码中的 img_path 变量。")
        return

    print(f"Testing image: {img_path}")
    
    # --- 2. 初始化模型与引擎 ---
    detector = ChamberDetector()
    engine = CrossGeometryEngine() # V10.0
    
    img = cv2.imread(img_path)
    if img is None:
        print("[Error] 无法读取图片")
        return

    # --- 3. 运行 YOLO 检测 ---
    points, classes = detector.detect(img)
    
    print(f"检测点数: {len(points)}")
    print(f"类别列表: {classes}")

    if len(points) == 0:
        print("[Error] YOLO 未检测到任何点！")
        return

    # --- 4. 可视化阶段 1: 原始 YOLO 结果 (Raw) ---
    # 目的：确认 YOLO 本身是否把空白腔识别成了 0 (BLANK)
    vis_raw = img.copy()
    for i, (pt, cls) in enumerate(zip(points, classes)):
        cx, cy = int(pt[0]), int(pt[1])
        
        if cls == config.CLASS_ID_BLANK: # Class 0
            color = (0, 0, 255) # Red
            thickness = 4
            text = f"BLANK({cls})"
            radius = 20
        else:
            color = (0, 255, 0) # Green
            thickness = 2
            text = str(cls)
            radius = 10
        
        cv2.circle(vis_raw, (cx, cy), radius, color, thickness)
        cv2.putText(vis_raw, text, (cx-10, cy-radius-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    save_path_raw = "check_01_yolo_raw.jpg"
    cv2.imwrite(save_path_raw, vis_raw)
    print(f"-> [Step 1] 原始 YOLO 结果已保存至: {save_path_raw}")
    print("   请检查：红圈是否准确套在了空白腔上？(验证 YOLO 准确性)")

    # --- 5. 可视化阶段 2: V10.0 对齐与切片结果 ---
    # 目的：确认经过刚性变换后，红圈是否在正上方，切片框是否跟随真实坐标
    if len(points) >= 12:
        print("正在运行 V10.0 对齐引擎...")
        patches, vis_aligned = engine.process(img, points, classes)
        
        if vis_aligned is not None:
            save_path_aligned = "check_02_aligned_v10.jpg"
            cv2.imwrite(save_path_aligned, vis_aligned)
            print(f"-> [Step 2] 对齐最终结果已保存至: {save_path_aligned}")
            print("   请检查：")
            print("   1. 图片是否已转正？(十字架端正)")
            print("   2. 红圈是否在正上方 (North)？")
            print("   3. 蓝色实心点是否完美落在腔室中心？")
            
            # 顺便检查一下切片形状
            print(f"   生成的切片数量: {len(patches)}")
            if len(patches) > 0:
                print(f"   单个切片尺寸: {patches[0].shape}")
        else:
            print("[Error] 对齐失败 (Points < 12 或逻辑错误)")
    else:
        print(f"[Skip] 点数不足 12 (Found {len(points)})，跳过对齐测试。")

if __name__ == "__main__":
    main()