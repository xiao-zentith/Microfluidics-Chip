import os
import shutil
import random
import math
from pathlib import Path

# --- 1. 配置您的路径 ---

# ！！注意：请将您所有（50张或500张）已标注好的图像和标签放在这里
SOURCE_IMAGES_DIR = Path("/home/asus515/PycharmProjects/YOLO_v11/dataset/dataset_all/images")
SOURCE_LABELS_DIR = Path("/home/asus515/PycharmProjects/YOLO_v11/dataset/dataset_all/labels")

# 这是脚本将要创建的、符合YOLOv8（`yolo_dataset_plan.md`）要求的最终文件夹
DEST_DATASET_DIR = Path("/home/asus515/PycharmProjects/YOLO_v11/dataset/yolo_dataset")

# ！！注意：这是您的划分比例
# 0.7 = 70% 训练集, 30% 验证集
TRAIN_RATIO = 0.7

# --- 2. 创建目标文件夹 (train/val) ---
# （`yolo_dataset_plan.md` 第2节的结构）
train_img_path = DEST_DATASET_DIR / "images" / "train"
val_img_path = DEST_DATASET_DIR / "images" / "val"
train_lbl_path = DEST_DATASET_DIR / "labels" / "train"
val_lbl_path = DEST_DATASET_DIR / "labels" / "val"

# exist_ok=True 意味着如果文件夹已存在,不会报错
print("正在创建目标文件夹...")
train_img_path.mkdir(parents=True, exist_ok=True)
val_img_path.mkdir(parents=True, exist_ok=True)
train_lbl_path.mkdir(parents=True, exist_ok=True)
val_lbl_path.mkdir(parents=True, exist_ok=True)

# --- 3. 查找所有文件并准备划分 ---
print(f"正在从 {SOURCE_IMAGES_DIR} 查找文件...")

# 查找所有合规的图像文件
image_extensions = ['.jpg', '.jpeg', '.png']
all_images = [
    f for f in SOURCE_IMAGES_DIR.iterdir() 
    if f.is_file() and f.suffix.lower() in image_extensions
]

if not all_images:
    print(f"*** 错误: 在 {SOURCE_IMAGES_DIR} 中未找到任何图像文件。请检查路径。")
    exit()

# 随机打乱列表（非常重要！）
random.shuffle(all_images)

# 计算分割点
total_count = len(all_images)
train_count = math.floor(total_count * TRAIN_RATIO)
val_count = total_count - train_count

train_files = all_images[:train_count]
val_files = all_images[train_count:]

print(f"总共找到 {total_count} 张图像。")
print(f"划分: {train_count} 张用于训练 (train), {val_count} 张用于验证 (val)。")

# --- 4. 定义文件移动函数 ---
def move_files(file_list, dest_img_path, dest_lbl_path):
    """
    一个辅助函数，用于移动图像和其对应的.txt标签文件。
    """
    moved_count = 0
    for img_file_path in file_list:
        # 1. 准备路径
        # (例如: .../chip_001.jpg)
        img_name = img_file_path.name
        
        # (例如: chip_001)
        base_name = img_file_path.stem 
        
        # (例如: chip_001.txt)
        lbl_name = base_name + ".txt"
        lbl_file_path = SOURCE_LABELS_DIR / lbl_name

        # 2. 检查标签是否存在
        if not lbl_file_path.exists():
            print(f"--- 警告: 图像 {img_name} 对应的标签 {lbl_name} 不存在，跳过此文件。")
            continue
            
        # 3. 移动文件（我们使用 'copy' 更安全，如果用 'rename' 会移动原始文件）
        # shutil.copy(str(img_file_path), str(dest_img_path / img_name))
        # shutil.copy(str(lbl_file_path), str(dest_lbl_path / lbl_name))
        
        # 使用 'rename' (移动) 更快，但会清空源文件夹
        img_file_path.rename(dest_img_path / img_name)
        lbl_file_path.rename(dest_lbl_path / lbl_name)
        
        moved_count += 1
    return moved_count

# --- 5. 执行划分 ---
print("\n正在处理 'train' 集...")
moved_train = move_files(train_files, train_img_path, train_lbl_path)

print(f"\n正在处理 'val' 集...")
moved_val = move_files(val_files, val_img_path, val_lbl_path)

print("\n--- 划分完成！ ---")
print(f"成功移动 {moved_train} 个训练文件 (图像+标签)。")
print(f"成功移动 {moved_val} 个验证文件 (图像+标签)。")
print(f"您的YOLOv11（`yolo_dataset_plan.md`）数据集已在 '{DEST_DATASET_DIR}' 文件夹中准备就绪。")