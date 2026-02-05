#!/usr/bin/env python3
"""
YOLO 数据集离线增强脚本 (光照/噪声专用) - v2.0

功能：
利用 stage2 定义的高级物理光照退化模型 (ISP Degradation)，
对 YOLO 训练集进行离线增强，以解决验证集光照/距离域偏移 (Domain Shift) 问题。

v2.0 新增特性：
- 分层采样：70% mild / 25% medium / 5% extreme (可配置)
- CLAHE 预处理开关 (默认开)
- Invert 预处理开关 (默认关)
- 增强档位日志记录

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
    # 基本用法
    python scripts/augment_yolo_dataset.py --input data/stage1_detection/yolo_v3/images/train
    
    # 自定义分层比例
    python scripts/augment_yolo_dataset.py --input ... --mild-ratio 0.6 --medium-ratio 0.3 --extreme-ratio 0.1
    
    # 启用 CLAHE + 禁用 Invert
    python scripts/augment_yolo_dataset.py --input ... --enable-clahe --no-invert
"""

import argparse
import random
import shutil
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Optional

# 添加 src 到路径以便导入模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microfluidics_chip.stage2_correction.augmentations import apply_isp_degradation
from microfluidics_chip.core.logger import setup_logger, get_logger

logger = get_logger("augment_yolo")


# ==================== 分层增强参数配置 ====================

@dataclass
class AugmentationTier:
    """增强档位参数"""
    name: str
    illum_strength: Tuple[float, float]
    wb_range: Tuple[float, float]
    exposure_range: Tuple[float, float]
    gamma_range: Tuple[float, float]
    peak_photon_range: Tuple[int, int]


# 三档增强参数
TIER_MILD = AugmentationTier(
    name="mild",
    illum_strength=(0.1, 0.3),
    wb_range=(0.95, 1.05),
    exposure_range=(0.9, 1.1),
    gamma_range=(0.9, 1.1),
    peak_photon_range=(60, 100)
)

TIER_MEDIUM = AugmentationTier(
    name="medium",
    illum_strength=(0.3, 0.5),
    wb_range=(0.85, 1.15),
    exposure_range=(0.75, 1.25),
    gamma_range=(0.8, 1.2),
    peak_photon_range=(40, 60)
)

TIER_EXTREME = AugmentationTier(
    name="extreme",
    illum_strength=(0.5, 0.8),
    wb_range=(0.75, 1.25),
    exposure_range=(0.5, 1.5),
    gamma_range=(0.7, 1.4),
    peak_photon_range=(20, 40)
)


def sample_tier(mild_ratio: float, medium_ratio: float, extreme_ratio: float) -> AugmentationTier:
    """
    根据比例随机采样增强档位
    
    :param mild_ratio: mild 档位比例
    :param medium_ratio: medium 档位比例
    :param extreme_ratio: extreme 档位比例
    :return: 采样到的档位
    """
    r = random.random()
    if r < mild_ratio:
        return TIER_MILD
    elif r < mild_ratio + medium_ratio:
        return TIER_MEDIUM
    else:
        return TIER_EXTREME


def apply_tiered_augmentation(
    image: np.ndarray,
    tier: AugmentationTier
) -> np.ndarray:
    """
    应用分层增强
    
    :param image: 输入图像 (BGR, uint8)
    :param tier: 增强档位
    :return: 增强后的图像
    """
    # 在档位范围内随机采样参数
    illum_strength = random.uniform(*tier.illum_strength)
    
    return apply_isp_degradation(
        image,
        illum_strength=illum_strength,
        wb_range=tier.wb_range,
        exposure_range=tier.exposure_range,
        gamma_range=tier.gamma_range,
        peak_photon_range=tier.peak_photon_range
    )


# ==================== 预处理函数 ====================

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    对图像应用 CLAHE (仅 L 通道)
    
    :param image: BGR 图像
    :param clip_limit: CLAHE clip limit
    :param tile_grid_size: 网格大小
    :return: CLAHE 增强后的图像
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_invert(image: np.ndarray) -> np.ndarray:
    """反转图像亮度"""
    return 255 - image


# ==================== 主增强函数 ====================

def augment_yolo_dataset(
    input_dir: Path,
    output_root: Path = None,
    multiplier: int = 5,
    p_augment: float = 1.0,
    # 分层采样比例
    mild_ratio: float = 0.70,
    medium_ratio: float = 0.25,
    extreme_ratio: float = 0.05,
    # 预处理开关
    enable_clahe: bool = True,
    enable_invert: bool = False,
    clahe_clip_limit: float = 2.0
):
    """
    增强 YOLO 数据集 (v2.0 分层采样版)
    
    :param input_dir: 输入图片目录 (images/train)
    :param output_root: 新数据集的根目录。如果为 None，则原地增强
    :param multiplier: 每张原图生成的增强副本数量
    :param p_augment: 每张图进行增强的概率
    :param mild_ratio: mild 档位比例 (默认 70%)
    :param medium_ratio: medium 档位比例 (默认 25%)
    :param extreme_ratio: extreme 档位比例 (默认 5%)
    :param enable_clahe: 是否启用 CLAHE 预处理
    :param enable_invert: 是否启用 Invert 预处理
    :param clahe_clip_limit: CLAHE clip limit
    """
    # 验证比例总和
    total_ratio = mild_ratio + medium_ratio + extreme_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.warning(f"Tier ratios sum to {total_ratio:.2f}, normalizing...")
        mild_ratio /= total_ratio
        medium_ratio /= total_ratio
        extreme_ratio /= total_ratio
    
    # 寻找标签目录
    dataset_root = input_dir.parent.parent
    image_dirname = input_dir.name
    label_dir = dataset_root / "labels" / image_dirname
    
    if not label_dir.exists():
        logger.error(f"Could not find label directory at {label_dir}")
        logger.error("Please ensure dataset follows standard YOLO structure: images/{split} and labels/{split}")
        return

    # 确定输出目录
    if output_root:
        output_image_dir = output_root / "images" / image_dirname
        output_label_dir = output_root / "labels" / image_dirname
        logger.info(f"Creating new dataset at: {output_root}")
    else:
        output_image_dir = input_dir
        output_label_dir = label_dir
        logger.info(f"Augmenting in-place at: {input_dir}")

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # 日志配置信息
    logger.info("=" * 50)
    logger.info("Augmentation Configuration (v2.0)")
    logger.info("=" * 50)
    logger.info(f"Input Images:   {input_dir}")
    logger.info(f"Input Labels:   {label_dir}")
    logger.info(f"Output Images:  {output_image_dir}")
    logger.info(f"Multiplier:     {multiplier}x")
    logger.info(f"Tier Ratios:    mild={mild_ratio:.0%}, medium={medium_ratio:.0%}, extreme={extreme_ratio:.0%}")
    logger.info(f"Preprocessing:  CLAHE={enable_clahe}, Invert={enable_invert}")
    logger.info("=" * 50)
    
    # 获取所有原始图片
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in extensions]
    original_images = [p for p in images if "_aug_" not in p.name]
    logger.info(f"Found {len(original_images)} original images")
    
    # 统计
    count_generated = 0
    count_copied = 0
    tier_counts = {"mild": 0, "medium": 0, "extreme": 0}

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
        
        # 4. 可选预处理
        if enable_clahe:
            img = apply_clahe(img, clip_limit=clahe_clip_limit)
        if enable_invert:
            img = apply_invert(img)
            
        # 5. 生成增强副本
        for i in range(multiplier):
            if random.random() > p_augment:
                continue
            
            # 采样增强档位
            tier = sample_tier(mild_ratio, medium_ratio, extreme_ratio)
            tier_counts[tier.name] += 1
                
            try:
                aug_img = apply_tiered_augmentation(img, tier)
            except Exception as e:
                logger.error(f"Augmentation failed for {img_path.name}: {e}")
                continue
            
            # 保存增强后的图片 (文件名包含档位标记)
            new_stem = f"{img_path.stem}_aug_{tier.name}_{i}"
            new_img_name = new_stem + img_path.suffix
            new_img_path = output_image_dir / new_img_name
            
            cv2.imwrite(str(new_img_path), aug_img)
            
            # 复制标签 (坐标不变)
            new_txt_path = output_label_dir / (new_stem + ".txt")
            shutil.copy2(txt_path, new_txt_path)
            
            count_generated += 1
            
    # 汇总日志
    logger.info("=" * 50)
    logger.info("Augmentation Complete!")
    logger.info("=" * 50)
    if output_root:
        logger.info(f"Copied {count_copied} original samples")
        
        # 复制 data.yaml
        yaml_src = dataset_root / "data.yaml"
        if yaml_src.exists():
            yaml_dst = output_root / "data.yaml"
            if not yaml_dst.exists():
                shutil.copy2(yaml_src, yaml_dst)
                logger.info(f"Copied data.yaml to {yaml_dst}")
    
    logger.info(f"Generated {count_generated} new augmented samples")
    logger.info(f"Tier distribution: {tier_counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment YOLO dataset with Physics-based ISP degradation (v2.0 Tiered Sampling)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 基本参数
    parser.add_argument("--input", type=Path, required=True, 
                        help="Path to images directory (e.g. data/stage1_detection/yolo_v3/images/train)")
    parser.add_argument("--output-root", type=Path, default=None, 
                        help="Root directory for the NEW dataset. If not set, augments in-place.")
    parser.add_argument("--multiplier", type=int, default=5, 
                        help="Number of augmented copies per image (default: 5)")
    parser.add_argument("--prob", type=float, default=1.0, 
                        help="Probability of augmentation (default: 1.0)")
    
    # 分层采样比例
    parser.add_argument("--mild-ratio", type=float, default=0.70,
                        help="Ratio of mild augmentation (default: 0.70)")
    parser.add_argument("--medium-ratio", type=float, default=0.25,
                        help="Ratio of medium augmentation (default: 0.25)")
    parser.add_argument("--extreme-ratio", type=float, default=0.05,
                        help="Ratio of extreme augmentation (default: 0.05)")
    
    # 预处理开关
    parser.add_argument("--enable-clahe", dest="enable_clahe", action="store_true",
                        help="Enable CLAHE preprocessing (default: enabled)")
    parser.add_argument("--no-clahe", dest="enable_clahe", action="store_false",
                        help="Disable CLAHE preprocessing")
    parser.set_defaults(enable_clahe=True)
    
    parser.add_argument("--enable-invert", dest="enable_invert", action="store_true",
                        help="Enable Invert preprocessing (default: disabled)")
    parser.add_argument("--no-invert", dest="enable_invert", action="store_false",
                        help="Disable Invert preprocessing")
    parser.set_defaults(enable_invert=False)
    
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0,
                        help="CLAHE clip limit (default: 2.0)")
    
    args = parser.parse_args()
    
    setup_logger(level="INFO")
    
    if not args.input.exists():
        print(f"Error: Input directory {args.input} does not exist.")
        return 1
        
    augment_yolo_dataset(
        input_dir=args.input,
        output_root=args.output_root,
        multiplier=args.multiplier,
        p_augment=args.prob,
        mild_ratio=args.mild_ratio,
        medium_ratio=args.medium_ratio,
        extreme_ratio=args.extreme_ratio,
        enable_clahe=args.enable_clahe,
        enable_invert=args.enable_invert,
        clahe_clip_limit=args.clahe_clip_limit
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
