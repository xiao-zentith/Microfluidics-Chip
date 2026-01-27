# config.py

import os

# --- 硬件与模型参数 ---
YOLO_WEIGHTS_PATH = "/home/asus515/PycharmProjects/YOLO_v11/ultralytics/runs/detect/chip_detection_run12/weights/best.pt" 

# YOLO 类别定义
CLASS_ID_BLANK = 0      # 空白腔 (锚点)
CLASS_ID_NON_BLANK = 1  # 非空白腔

# --- 几何与图像参数 (核心修改) ---
CANVAS_SIZE = 600       # 理想画布大小 (600x600)
SLICE_SIZE = (80, 80)   # 切片大小

# 理想十字布局参数 (像素单位，基于600x600画布)
# 调整这些值可以改变切片在画布上的疏密
IDEAL_CENTER_GAP = 60   # 第一圈腔室距离中心的距离
IDEAL_CHAMBER_STEP = 50 # 同一旋臂上腔室的间距

# 切片半径 (用于在理想画布上裁剪)
# 建议设为 STEP 的一半左右，避免重叠
CROP_RADIUS = 25       

RANSAC_THRESH = 5.0

# --- 路径配置 ---
INPUT_RAW_ROOT = "/home/asus515/PycharmProjects/YOLO_v11/dataset/preprocess/raw_data" 
GT_LIBRARY_DIR = "/home/asus515/PycharmProjects/YOLO_v11/dataset/preprocess/gt_library"
OUTPUT_DIR = "/home/asus515/PycharmProjects/YOLO_v11/preprocess_result/processed_dataset"
DEBUG_DIR = "/home/asus515/PycharmProjects/YOLO_v11/preprocess_result/debug_vis"