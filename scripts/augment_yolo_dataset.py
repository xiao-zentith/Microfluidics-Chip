"""
YOLO 数据集离线增强脚本 (光照/噪声专用)

功能：
利用 stage2 定义的高级物理光照退化模型 (ISP Degradation)，
对 YOLO 训练集进行离线增强，以解决验证集光照/距离域偏移 (Domain Shift) 问题。

增强内容：
- 光照场 (Illumination Field): 渐晕、单侧光
- 白平衡漂移 (White Balance)
- 曝光变化 (Exposure)
- Gamma 校正
- 光子噪声 (Shot Noise)

注意：
此脚本**不进行**几何变换 (旋转/缩放/平移)，亦不改变标签坐标。
几何变换应交由 YOLO 训练时的在线增强 (degrees, scale, mosaic) 处理。

用法：
    python scripts/augment_yolo_dataset.py --input data/stage1_detection/yolo_v1/images/train
"""

import argparse
import random
import shutil
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加 src 到路径以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microfluidics_chip.stage2_correction.augmentations import apply_isp_degradation
from microfluidics_chip.core.logger import setup_logger, get_logger

logger = get_logger("augment_yolo")


def augment_yolo_dataset(
    input_dir: Path,
    output_root: Path = None,
    multiplier: int = 5,
    p_augment: float = 1.0
):
    """
    增强 YOLO 数据集
    
    :param input_dir: 输入图片目录 (images/train)
    :param output_root: 新数据集的根目录 (e.g. data/stage1_detection/yolo_v1_aug). 
                        如果为None，则在原目录原地增强。
    :param multiplier: 每张原图生成的增强副本数量
    :param p_augment: 每张图进行增强的概率
    """
    # 寻找标签目录 (假设遵循 YOLO 标准结构: images/train -> labels/train)
    dataset_root = input_dir.parent.parent
    image_dirname = input_dir.name  # e.g., 'train' or 'val'
    label_dir = dataset_root / "labels" / image_dirname
    
    if not label_dir.exists():
        logger.error(f"Could not find label directory at {label_dir}")
        logger.error("Please ensure dataset follows standard YOLO structure: images/{split} and labels/{split}")
        return

    # 确定输出目录
    if output_root:
        # 创建新的独立数据集
        output_image_dir = output_root / "images" / image_dirname
        output_label_dir = output_root / "labels" / image_dirname
        logger.info(f"Creating new dataset at: {output_root}")
    else:
        # 原地修改
        output_image_dir = input_dir
        output_label_dir = label_dir
        logger.info(f"Augmenting in-place at: {input_dir}")

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input Images: {input_dir}")
    logger.info(f"Input Labels: {label_dir}")
    logger.info(f"Output Images: {output_image_dir}")
    logger.info(f"Multiplier:   {multiplier}x")
    
    # 获取所有图片
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in extensions]
    
    # 过滤掉已经是增强过的图片 (避免重复增强)
    original_images = [p for p in images if "_aug_" not in p.name]
    logger.info(f"Found {len(original_images)} original images")
    
    count_generated = 0
    count_copied = 0

    for img_path in tqdm(original_images, desc="Processing"):
        # 1. 检查标签是否存在
        txt_name = img_path.stem + ".txt"
        txt_path = label_dir / txt_name
        
        if not txt_path.exists():
            logger.warning(f"Label not found for {img_path.name}, skipping...")
            continue
            
        # 2. 如果是新数据集模式，先复制原始文件
        if output_root:
            out_img_path = output_image_dir / img_path.name
            out_txt_path = output_label_dir / txt_name
            
            if not out_img_path.exists():
                shutil.copy2(img_path, out_img_path)
            if not out_txt_path.exists():
                shutil.copy2(txt_path, out_txt_path)
            count_copied += 1

        # 3. 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to read {img_path}, skipping...")
            continue
            
        # 4. 生成增强副本
        for i in range(multiplier):
            if random.random() > p_augment:
                continue
                
            try:
                aug_img = apply_isp_degradation(
                    img, 
                    illum_strength=0.5,
                    wb_range=(0.8, 1.2),
                    exposure_range=(0.7, 1.3),
                    peak_photon_range=(20, 80)
                )
            except Exception as e:
                logger.error(f"Augmentation failed for {img_path.name}: {e}")
                continue
            
            # 保存增强后的图片
            new_stem = f"{img_path.stem}_aug_{i}"
            new_img_name = new_stem + img_path.suffix
            new_img_path = output_image_dir / new_img_name
            
            cv2.imwrite(str(new_img_path), aug_img)
            
            # 复制标签 (坐标不变)
            new_txt_path = output_label_dir / (new_stem + ".txt")
            shutil.copy2(txt_path, new_txt_path)
            
            count_generated += 1
            
    logger.info(f"Done!")
    if output_root:
        logger.info(f"Copied {count_copied} original samples.")
        
        # 尝试复制 data.yaml
        yaml_src = dataset_root / "data.yaml"
        if yaml_src.exists():
            yaml_dst = output_root / "data.yaml"
            if not yaml_dst.exists():
                shutil.copy2(yaml_src, yaml_dst)
                logger.info(f"Copied data.yaml to {yaml_dst}")
                logger.warning("NOTE: You may need to update 'path' or 'train/val' paths in the new data.yaml if they are absolute.")
    
    logger.info(f"Generated {count_generated} new augmented samples.")


def main():
    parser = argparse.ArgumentParser(description="Augment YOLO dataset with Physics-based ISP degradation")
    parser.add_argument("--input", type=Path, required=True, help="Path to images directory (e.g. data/stage1_detection/yolo_v1/images/train)")
    parser.add_argument("--output-root", type=Path, default=None, help="Root directory for the NEW dataset (e.g. data/stage1_detection/yolo_v1_aug). If not set, augments in-place.")
    parser.add_argument("--multiplier", type=int, default=5, help="Number of augmented copies per image")
    parser.add_argument("--prob", type=float, default=1.0, help="Probability of augmentation")
    
    args = parser.parse_args()
    
    setup_logger(level="INFO")
    
    if not args.input.exists():
        print(f"Error: Input directory {args.input} does not exist.")
        return 1
        
    augment_yolo_dataset(
        input_dir=args.input,
        output_root=args.output_root,
        multiplier=args.multiplier,
        p_augment=args.prob
    )
    return 0


if __name__ == "__main__":
    exit(main())
