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
from datetime import datetime

# 添加 src 到路径以便导入模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microfluidics_chip.stage2_correction.augmentations import apply_isp_degradation
from microfluidics_chip.core.logger import setup_logger, get_logger

logger = get_logger("augment_yolo")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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
    """根据比例随机采样增强档位"""
    r = random.random()
    if r < mild_ratio:
        return TIER_MILD
    elif r < mild_ratio + medium_ratio:
        return TIER_MEDIUM
    else:
        return TIER_EXTREME


# ==================== 显式参数采样 (v2.2 新增) ====================

@dataclass
class DegradationParams:
    """退化参数（单值，用于日志记录）"""
    illum_strength: float
    wb_r: float
    wb_b: float
    exposure: float
    gamma: float
    peak_photon: int
    tier_name: str = ""
    
    def to_log_str(self) -> str:
        """格式化为日志字符串"""
        return (f"tier={self.tier_name}, illum={self.illum_strength:.2f}, "
                f"wb_r={self.wb_r:.2f}, wb_b={self.wb_b:.2f}, "
                f"exp={self.exposure:.2f}, gamma={self.gamma:.2f}, "
                f"photon={self.peak_photon}")


def sample_degradation_params(tier: AugmentationTier) -> DegradationParams:
    """
    显式采样退化参数（v2.2）
    
    对每个增强副本独立采样具体数值，便于日志记录和复现
    """
    return DegradationParams(
        illum_strength=random.uniform(*tier.illum_strength),
        wb_r=random.uniform(*tier.wb_range),
        wb_b=random.uniform(*tier.wb_range),
        exposure=random.uniform(*tier.exposure_range),
        gamma=random.uniform(*tier.gamma_range),
        peak_photon=random.randint(*tier.peak_photon_range),
        tier_name=tier.name
    )


def apply_isp_with_params(
    image: np.ndarray,
    params: DegradationParams
) -> np.ndarray:
    """
    应用 ISP 退化（使用显式单值参数）
    
    这是对 apply_isp_degradation 的 wrapper，接受单值参数而非 range
    内部复制了核心逻辑以避免重复随机采样
    """
    # 统一转换为 float32 [0, 1]
    is_uint8 = image.dtype == np.uint8
    if is_uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.copy()
    
    h, w = img.shape[:2]
    
    # 1. 空间光照场 M_illum
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    angle = np.deg2rad(random.uniform(0, 360))
    
    gradient = X * np.cos(angle) + Y * np.sin(angle)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    
    cx, cy = w // 2, h // 2
    radial = np.sqrt((X - cx)**2 + (Y - cy)**2)
    radial = radial / (radial.max() + 1e-6)
    
    mix = random.random()
    field = mix * gradient + (1 - mix) * radial
    illum_map = 1.0 - params.illum_strength * field
    img = img * illum_map[..., np.newaxis]
    
    # 2. 白平衡漂移 (使用显式采样的 wb_r, wb_b)
    img[:, :, 2] *= params.wb_r  # R通道 (BGR格式)
    img[:, :, 0] *= params.wb_b  # B通道
    
    # 3. 曝光增益
    img = img * params.exposure
    
    # 4. Gamma校正
    img = np.clip(img, 0, 1)
    img = np.power(img, params.gamma)
    
    # 5. Shot Noise
    from microfluidics_chip.stage2_correction.augmentations import apply_shot_noise
    img = apply_shot_noise(img, params.peak_photon)
    
    # 转换回原始格式
    if is_uint8:
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        return np.clip(img, 0, 1).astype(np.float32)


# ==================== 预处理函数 (v2.2 统一模块) ====================

# CLAHE 参数随机范围
CLAHE_CLIP_RANGE = (1.5, 2.5)
CLAHE_GRID_OPTIONS = [(4, 4), (8, 8)]


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """对图像应用 CLAHE (仅 L 通道)"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_invert(image: np.ndarray) -> np.ndarray:
    """反转图像亮度"""
    return 255 - image


def preprocess_image(
    image: np.ndarray,
    enable_clahe: bool = True,
    enable_invert: bool = False,
    invert_prob: float = 0.0,
    clahe_clip_limit: Optional[float] = None,
    clahe_grid_size: Optional[Tuple[int, int]] = None,
    randomize_clahe: bool = True
) -> np.ndarray:
    """
    统一预处理函数 (v2.2)
    
    可用于训练增强和推理预处理，确保一致性
    
    :param image: 输入图像 (BGR, uint8)
    :param enable_clahe: 是否启用 CLAHE
    :param enable_invert: 是否启用 Invert（作为开关，受 invert_prob 控制）
    :param invert_prob: Invert 触发概率（默认 0.0，即使 enable_invert=True 也不必反相）
    :param clahe_clip_limit: CLAHE clip limit（None 则随机）
    :param clahe_grid_size: CLAHE grid size（None 则随机）
    :param randomize_clahe: 是否随机化 CLAHE 参数
    :return: 预处理后的图像
    """
    result = image.copy()
    
    # Invert (概率触发)
    if enable_invert and random.random() < invert_prob:
        result = apply_invert(result)
    
    # CLAHE
    if enable_clahe:
        if randomize_clahe:
            clip = clahe_clip_limit if clahe_clip_limit else random.uniform(*CLAHE_CLIP_RANGE)
            grid = clahe_grid_size if clahe_grid_size else random.choice(CLAHE_GRID_OPTIONS)
        else:
            clip = clahe_clip_limit if clahe_clip_limit else 2.0
            grid = clahe_grid_size if clahe_grid_size else (8, 8)
        result = apply_clahe(result, clip_limit=clip, tile_grid_size=grid)
    
    return result


# ==================== 质量控制 (v2.2 新增) ====================

def quality_check(
    image: np.ndarray,
    min_std: float = 10.0,
    min_mean: float = 20.0,
    max_mean: float = 240.0
) -> bool:
    """
    检查增强后的图像质量
    
    避免 extreme 档位产生"满屏噪点"或过暗/过亮的不可用样本
    
    :param image: 增强后的图像
    :param min_std: 最小标准差阈值
    :param min_mean: 最小平均亮度
    :param max_mean: 最大平均亮度
    :return: True 表示通过质量检查
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    if std_val < min_std:
        return False
    if mean_val < min_mean or mean_val > max_mean:
        return False
    
    return True


def apply_tiered_augmentation_v2(
    image: np.ndarray,
    tier: AugmentationTier,
    enable_clahe: bool = True,
    enable_invert: bool = False,
    invert_prob: float = 0.0,
    clahe_position: str = "after_degradation",
    max_retries: int = 3
) -> Tuple[np.ndarray, DegradationParams]:
    """
    应用分层增强 (v2.2)
    
    :param image: 输入图像 (BGR, uint8)
    :param tier: 增强档位
    :param enable_clahe: 是否启用 CLAHE
    :param enable_invert: 是否启用 Invert
    :param invert_prob: Invert 触发概率
    :param clahe_position: CLAHE 位置 ("before_degradation" 或 "after_degradation")
    :param max_retries: 质量检查失败时的重采样次数
    :return: (增强后的图像, 采样参数)
    """
    for attempt in range(max_retries):
        # 1. 显式采样参数
        params = sample_degradation_params(tier)
        
        # 2. 预处理 (before_degradation)
        if clahe_position == "before_degradation":
            processed = preprocess_image(
                image,
                enable_clahe=enable_clahe,
                enable_invert=enable_invert,
                invert_prob=invert_prob,
                randomize_clahe=True
            )
        else:
            processed = image.copy()
        
        # 3. 应用 ISP 退化
        degraded = apply_isp_with_params(processed, params)
        
        # 4. 后处理 (after_degradation，默认)
        if clahe_position == "after_degradation":
            result = preprocess_image(
                degraded,
                enable_clahe=enable_clahe,
                enable_invert=enable_invert,
                invert_prob=invert_prob,
                randomize_clahe=True
            )
        else:
            result = degraded
        
        # 5. 质量检查
        if quality_check(result):
            return result, params
        
        # 质量不合格：extreme 档降级到 medium
        if tier == TIER_EXTREME and attempt < max_retries - 1:
            logger.debug(f"Quality check failed, downgrading from extreme to medium (attempt {attempt + 1})")
            tier = TIER_MEDIUM
    
    # 最后一次尝试，不再检查质量
    return result, params


def _is_image_file(path: Path) -> bool:
    """判断是否为支持的图像文件。"""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _ensure_split_labels(src_img_dir: Path, dst_label_dir: Path) -> int:
    """
    为 split 下每张图像创建空标签文件（当源 labels/split 不存在时使用）。

    :return: 新建的空标签数量
    """
    created = 0
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    for img_path in src_img_dir.iterdir():
        if not _is_image_file(img_path):
            continue
        txt_path = dst_label_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            txt_path.touch()
            created += 1
    return created


def _sync_other_splits(dataset_root: Path, output_root: Path, processed_split: str) -> None:
    """
    同步除 processed_split 外的 split（如 val/test）到新数据集。
    """
    src_images_root = dataset_root / "images"
    src_labels_root = dataset_root / "labels"

    if not src_images_root.exists():
        return

    synced = []
    generated_empty = 0
    for split_dir in sorted(src_images_root.iterdir()):
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name
        if split_name == processed_split:
            continue

        dst_img_dir = output_root / "images" / split_name
        dst_label_dir = output_root / "labels" / split_name
        shutil.copytree(split_dir, dst_img_dir, dirs_exist_ok=True)

        src_label_dir = src_labels_root / split_name
        if src_label_dir.exists():
            shutil.copytree(src_label_dir, dst_label_dir, dirs_exist_ok=True)
        else:
            generated_empty += _ensure_split_labels(split_dir, dst_label_dir)

        synced.append(split_name)

    if synced:
        logger.info(f"Synced additional splits: {synced}")
    if generated_empty > 0:
        logger.warning(
            f"Source labels missing for some splits, generated {generated_empty} empty label files."
        )


# ==================== 主增强函数 ====================

def augment_yolo_dataset(
    input_dir: Path,
    output_root: Path = None,
    multiplier: int = 3,  # v2.2: 从 5 改为 3
    p_augment: float = 0.7,  # v2.2: 从 1.0 改为 0.7
    # 分层采样比例
    mild_ratio: float = 0.70,
    medium_ratio: float = 0.25,
    extreme_ratio: float = 0.05,
    # 预处理参数 (v2.2)
    enable_clahe: bool = True,
    enable_invert: bool = False,
    invert_prob: float = 0.02,  # v2.2: 概率触发
    clahe_position: str = "after_degradation",  # v2.2: CLAHE 位置控制
    # 是否同步其他 split（val/test）
    sync_other_splits: bool = False,
    # 原地修改开关
    in_place: bool = False,
    # 日志详细程度
    verbose: bool = False
):
    """
    增强 YOLO 数据集 (v2.2 真实感 + 训练推理一致性)
    
    改进点:
    - CLAHE 位置可选 (before/after degradation)
    - Invert 概率触发
    - 每个副本独立预处理
    - 显式参数采样 + 日志记录
    - 质量控制机制
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
    if in_place:
        if output_root is not None:
            logger.warning("Argument --output-root is ignored because --in-place is set!")
        output_root = dataset_root
        output_image_dir = input_dir
        output_label_dir = label_dir
        logger.warning(f"⚠️  AUGMENTING IN-PLACE at: {input_dir}")
    else:
        if output_root is None:
            output_root = dataset_root.parent / f"{dataset_root.name}_augmented"
        output_image_dir = output_root / "images" / image_dirname
        output_label_dir = output_root / "labels" / image_dirname
        logger.info(f"Creating NEW dataset at: {output_root}")

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # 日志配置信息
    logger.info("=" * 60)
    logger.info("Augmentation Configuration (v2.2)")
    logger.info("=" * 60)
    logger.info(f"Input Images:       {input_dir}")
    logger.info(f"Output Root:        {output_root}")
    logger.info(f"Multiplier:         {multiplier}x (prob={p_augment:.0%})")
    logger.info(f"Tier Ratios:        mild={mild_ratio:.0%}, medium={medium_ratio:.0%}, extreme={extreme_ratio:.0%}")
    logger.info(f"CLAHE:              enabled={enable_clahe}, position={clahe_position}")
    logger.info(f"Invert:             enabled={enable_invert}, prob={invert_prob:.1%}")
    logger.info("=" * 60)
    
    # 获取所有原始图片
    images = [p for p in input_dir.iterdir() if _is_image_file(p)]
    original_images = [p for p in images if "_aug_" not in p.name]
    logger.info(f"Found {len(original_images)} original images")
    
    # 统计
    count_generated = 0
    count_copied = 0
    count_quality_failures = 0
    tier_counts = {"mild": 0, "medium": 0, "extreme": 0}

    for img_path in tqdm(original_images, desc="Processing"):
        # 1. 检查标签是否存在
        txt_name = img_path.stem + ".txt"
        txt_path = label_dir / txt_name
        
        if not txt_path.exists():
            logger.warning(f"Label not found for {img_path.name}, skipping...")
            continue
            
        # 2. 如果是新数据集模式，先复制原始文件
        if not in_place:
            out_img_path = output_image_dir / img_path.name
            out_txt_path = output_label_dir / txt_name
            
            if not out_img_path.exists():
                shutil.copy2(img_path, out_img_path)
            if not out_txt_path.exists():
                shutil.copy2(txt_path, out_txt_path)
            count_copied += 1

        # 3. 读取原图 (不在此处预处理，每个副本独立处理)
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to read {img_path}, skipping...")
            continue
            
        # 4. 生成增强副本 (每个副本独立预处理 + 增强)
        for i in range(multiplier):
            if random.random() > p_augment:
                continue
            
            # 采样增强档位
            tier = sample_tier(mild_ratio, medium_ratio, extreme_ratio)
                
            try:
                # v2.2: 使用新的增强函数，每个副本独立预处理
                aug_img, params = apply_tiered_augmentation_v2(
                    img,
                    tier,
                    enable_clahe=enable_clahe,
                    enable_invert=enable_invert,
                    invert_prob=invert_prob,
                    clahe_position=clahe_position,
                    max_retries=3
                )
                
                # 记录实际使用的档位 (可能因质量控制降级)
                tier_counts[params.tier_name] += 1
                
                # 详细日志
                if verbose:
                    logger.debug(f"{img_path.name}[{i}]: {params.to_log_str()}")
                    
            except Exception as e:
                logger.error(f"Augmentation failed for {img_path.name}: {e}")
                continue
            
            # 保存增强后的图片
            new_stem = f"{img_path.stem}_aug_{params.tier_name}_{i}"
            new_img_name = new_stem + img_path.suffix
            new_img_path = output_image_dir / new_img_name
            
            cv2.imwrite(str(new_img_path), aug_img)
            
            # 复制标签 (坐标不变)
            new_txt_path = output_label_dir / (new_stem + ".txt")
            shutil.copy2(txt_path, new_txt_path)
            
            count_generated += 1
            
    # 汇总日志
    logger.info("=" * 60)
    logger.info("Augmentation Complete!")
    logger.info("=" * 60)
    if not in_place:
        logger.info(f"Copied {count_copied} original samples")

        # 可选：同步其他 split（如 val/test）
        if sync_other_splits:
            _sync_other_splits(
                dataset_root=dataset_root,
                output_root=output_root,
                processed_split=image_dirname
            )
        
        # 复制 data.yaml
        yaml_src = dataset_root / "data.yaml"
        if yaml_src.exists():
            yaml_dst = output_root / "data.yaml"
            if not yaml_dst.exists():
                shutil.copy2(yaml_src, yaml_dst)
                logger.info(f"Copied data.yaml to {yaml_dst}")
            
            try:
                with open(yaml_dst, 'a') as f:
                    f.write(f"\n# Augmented at {datetime.now().isoformat()}\n")
                    f.write(f"# Source: {dataset_root.name}\n")
                    f.write(f"# Config: multiplier={multiplier}, prob={p_augment}, clahe_position={clahe_position}\n")
            except Exception:
                pass
    
    logger.info(f"Generated {count_generated} new augmented samples")
    logger.info(f"Tier distribution: {tier_counts}")
    logger.info(f"New dataset location: {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment YOLO dataset with Physics-based ISP degradation (v2.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Default: multiplier=3, prob=0.7, CLAHE after degradation
  python augment_yolo_dataset.py --input data/yolo_v3/images/train

  # Custom configuration
  python augment_yolo_dataset.py --input data/yolo_v3/images/train \\
      --multiplier 2 --prob 0.8 \\
      --clahe-position before_degradation \\
      --invert-prob 0.05 --verbose
"""
    )
    
    # 基本参数
    parser.add_argument("--input", type=Path, required=True, 
                        help="Path to images directory (e.g. data/stage1_detection/yolo_v3/images/train)")
    parser.add_argument("--output-root", type=Path, default=None, 
                        help="Root directory for the NEW dataset. Defaults to '{input_parent}_augmented'.")
    parser.add_argument("--sync-splits", action="store_true",
                        help="Also copy other splits (e.g., val/test) to output dataset.")
    parser.add_argument("--in-place", action="store_true",
                        help="WARNING: Augment in-place (modify original dataset). Overrides --output-root.")
    
    parser.add_argument("--multiplier", type=int, default=3, 
                        help="Number of augmented copies per image (default: 3)")
    parser.add_argument("--prob", type=float, default=0.7, 
                        help="Probability of augmentation per copy (default: 0.7)")
    
    # 分层采样比例
    parser.add_argument("--mild-ratio", type=float, default=0.70,
                        help="Ratio of mild augmentation (default: 0.70)")
    parser.add_argument("--medium-ratio", type=float, default=0.25,
                        help="Ratio of medium augmentation (default: 0.25)")
    parser.add_argument("--extreme-ratio", type=float, default=0.05,
                        help="Ratio of extreme augmentation (default: 0.05)")
    
    # CLAHE 控制
    parser.add_argument("--enable-clahe", dest="enable_clahe", action="store_true",
                        help="Enable CLAHE preprocessing (default: enabled)")
    parser.add_argument("--no-clahe", dest="enable_clahe", action="store_false",
                        help="Disable CLAHE preprocessing")
    parser.set_defaults(enable_clahe=True)
    
    parser.add_argument("--clahe-position", type=str, default="after_degradation",
                        choices=["before_degradation", "after_degradation"],
                        help="CLAHE position: 'before_degradation' or 'after_degradation' (default: after_degradation)")
    
    # Invert 控制
    parser.add_argument("--enable-invert", dest="enable_invert", action="store_true",
                        help="Enable Invert preprocessing (default: disabled)")
    parser.add_argument("--no-invert", dest="enable_invert", action="store_false",
                        help="Disable Invert preprocessing")
    parser.set_defaults(enable_invert=False)
    
    parser.add_argument("--invert-prob", type=float, default=0.02,
                        help="Probability of Invert per sample when enabled (default: 0.02)")
    
    # 其他
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging (log each sample's parameters)")
    
    args = parser.parse_args()
    
    setup_logger(level="DEBUG" if args.verbose else "INFO")
    
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
        invert_prob=args.invert_prob,
        clahe_position=args.clahe_position,
        sync_other_splits=args.sync_splits,
        in_place=args.in_place,
        verbose=args.verbose
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
