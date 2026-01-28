"""
训练数据准备脚本

处理数据集结构：
  dataset/
    ├── chip001/
    │   ├── gt.png          # 理想图（Ground Truth）
    │   ├── dirty_01.png    # 受干扰图1
    │   ├── dirty_02.png    # 受干扰图2
    │   └── ...
    └── chip002/
        └── ...

输出：
  - 切片数据配对（Dirty切片 ↔ GT切片）
  - 3个基准腔室切片（用于UNet双流输入）
  - NPZ格式训练数据

使用方法：
  python scripts/prepare_training_data.py dataset/training -o data/training.npz
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microfluidics_chip.core.config import get_default_config
from microfluidics_chip.stage1_detection.detector import ChamberDetector
from microfluidics_chip.stage1_detection.geometry_engine import CrossGeometryEngine
from microfluidics_chip.core.logger import get_logger, setup_logger

logger = get_logger("prepare_training_data")


def process_chip_directory(
    chip_dir: Path,
    detector: ChamberDetector,
    config
) -> list:
    """
    处理单个芯片目录
    
    :param chip_dir: 芯片目录（包含 gt.png 和多个 dirty_*.png）
    :param detector: YOLO检测器
    :param config: Stage1配置
    :return: 训练样本列表 [{'signal': ..., 'reference': ..., 'target': ...}, ...]
    """
    # 查找 GT 图像
    gt_candidates = list(chip_dir.glob("gt.*")) + list(chip_dir.glob("GT.*"))
    if not gt_candidates:
        logger.warning(f"No GT image found in {chip_dir}")
        return []
    
    gt_path = gt_candidates[0]
    
    # 查找 Dirty 图像
    dirty_patterns = ["dirty_*.png", "dirty_*.jpg", "noisy_*.png", "noisy_*.jpg"]
    dirty_paths = []
    for pattern in dirty_patterns:
        dirty_paths.extend(chip_dir.glob(pattern))
    
    if not dirty_paths:
        logger.warning(f"No dirty images found in {chip_dir}")
        return []
    
    logger.info(f"Processing {chip_dir.name}: 1 GT + {len(dirty_paths)} dirty images")
    
    # ==================== 处理 GT 图像 ====================
    gt_image = cv2.imread(str(gt_path))
    if gt_image is None:
        logger.error(f"Failed to read GT image: {gt_path}")
        return []
    
    # 检测 GT
    detections_gt = detector.detect(gt_image)
    if len(detections_gt) < 12:
        logger.error(f"Insufficient GT detections: {len(detections_gt)} < 12")
        return []
    
    # 几何变换 GT（使用独立引擎）
    gt_engine = CrossGeometryEngine(config.geometry)
    _, gt_slices, _, gt_debug = gt_engine.process(gt_image, detections_gt)
    
    if gt_slices is None or len(gt_slices) != 12:
        logger.error(f"Failed to process GT geometry")
        return []
    
    # 保存 GT 调试可视化（可选）
    if gt_debug is not None:
        debug_path = chip_dir / "debug_gt.png"
        cv2.imwrite(str(debug_path), gt_debug)
    
    # ==================== 提取基准腔室（可配置） ====================
    # 从配置读取基准腔室索引
    ref_indices = config.stage2.reference_chambers if hasattr(config, 'stage2') else [0, 1, 2]
    ref_mode = config.stage2.reference_mode if hasattr(config, 'stage2') else "stack"
    
    reference_slices = []
    for idx in ref_indices:
        if idx < len(gt_slices):
            reference_slices.append(gt_slices[idx].astype(np.float32) / 255.0)
    
    if not reference_slices:
        logger.error(f"No valid reference chambers found for indices {ref_indices}")
        return []
    
    logger.info(f"Using {len(reference_slices)} reference chambers: indices {ref_indices}, mode={ref_mode}")
    
    # 根据模式组合基准腔室
    if ref_mode == "average":
        # 平均模式：所有基准腔室取平均
        reference_combined = np.mean(reference_slices, axis=0, keepdims=True)  # (1, H, W, 3)
    elif ref_mode == "stack":
        # 堆叠模式：保持独立（默认）
        reference_combined = np.stack(reference_slices, axis=0)  # (N_ref, H, W, 3)
    else:
        # concat 或其他模式可在此扩展
        reference_combined = np.stack(reference_slices, axis=0)
    
    # ==================== 处理每个 Dirty 图像 ====================
    training_samples = []
    
    # 确定训练腔室范围（跳过用作reference的腔室）
    max_ref_idx = max(ref_indices) if ref_indices else -1
    training_start_idx = max_ref_idx + 1
    
    for dirty_path in tqdm(dirty_paths, desc=f"{chip_dir.name}", leave=False):
        dirty_image = cv2.imread(str(dirty_path))
        if dirty_image is None:
            logger.warning(f"Failed to read: {dirty_path}")
            continue
        
        # 检测 Dirty
        detections_dirty = detector.detect(dirty_image)
        if len(detections_dirty) < 12:
            logger.warning(f"Insufficient detections in {dirty_path.name}: {len(detections_dirty)}")
            continue
        
        # 几何变换 Dirty（使用独立引擎）
        dirty_engine = CrossGeometryEngine(config.stage1.geometry)
        _, dirty_slices, _, dirty_debug = dirty_engine.process(dirty_image, detections_dirty)
        
        if dirty_slices is None or len(dirty_slices) != 12:
            logger.warning(f"Failed to process {dirty_path.name}")
            continue
        
        # 保存 Dirty 调试可视化（可选）
        if dirty_debug is not None:
            debug_path = chip_dir / f"debug_{dirty_path.stem}.png"
            cv2.imwrite(str(debug_path), dirty_debug)
        
        # ==================== 配对：每个训练腔室生成一条数据 ====================
        for chamber_idx in range(training_start_idx, 12):
            signal_slice = dirty_slices[chamber_idx].astype(np.float32) / 255.0  # 干扰信号
            target_slice = gt_slices[chamber_idx].astype(np.float32) / 255.0      # 目标GT
            
            # 组装训练样本
            training_samples.append({
                'signal': signal_slice,
                'reference': reference_combined,  # 根据mode已组合
                'target': target_slice
            })
    
    logger.info(f"✓ {chip_dir.name}: Generated {len(training_samples)} training samples")
    return training_samples


def prepare_training_data(
    dataset_dir: Path,
    output_path: Path,
    config_path: Path = None,
    save_debug: bool = True
):
    """
    批量准备训练数据
    
    :param dataset_dir: 数据集根目录
    :param output_path: 输出NPZ文件路径
    :param config_path: 配置文件路径
    :param save_debug: 是否保存调试可视化
    """
    setup_logger(level="INFO")
    
    # 加载配置
    if config_path and config_path.exists():
        from microfluidics_chip.core.config import load_config_from_yaml
        config = load_config_from_yaml(config_path)
    else:
        config = get_default_config()
    
    # 初始化检测器
    logger.info("Initializing YOLO detector...")
    detector = ChamberDetector(config.stage1.yolo)
    logger.info("✓ Detector initialized")
    
    # 收集所有芯片目录
    chip_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(chip_dirs)} chip directories")
    
    # 处理每个芯片目录
    all_samples = []
    for chip_dir in tqdm(chip_dirs, desc="Processing chips"):
        samples = process_chip_directory(chip_dir, detector, config)  # 传递完整config
        all_samples.extend(samples)
    
    if not all_samples:
        logger.error("No training samples generated!")
        return
    
    # ==================== 转换为NumPy数组 ====================
    logger.info(f"Converting {len(all_samples)} samples to NumPy arrays...")
    
    signals = np.array([s['signal'] for s in all_samples], dtype=np.float32)
    references = np.array([s['reference'] for s in all_samples], dtype=np.float32)
    targets = np.array([s['target'] for s in all_samples], dtype=np.float32)
    
    logger.info(f"Signal shape:    {signals.shape}")      # (N, H, W, 3)
    logger.info(f"Reference shape: {references.shape}")    # (N, 3, H, W, 3)
    logger.info(f"Target shape:    {targets.shape}")       # (N, H, W, 3)
    
    # ==================== 保存到NPZ ====================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        signals=signals,
        references=references,
        targets=targets
    )
    
    logger.info(f"✓ Training data saved to: {output_path}")
    logger.info(f"Total samples: {len(all_samples)}")
    
    # 统计信息
    logger.info("=" * 60)
    logger.info("Dataset Statistics:")
    logger.info(f"  - Total chips: {len(chip_dirs)}")
    logger.info(f"  - Total samples: {len(all_samples)}")
    logger.info(f"  - Avg samples/chip: {len(all_samples) / len(chip_dirs):.1f}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from GT + Dirty images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="数据集根目录（包含多个芯片子目录）"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/training.npz"),
        help="输出NPZ文件路径（默认: data/training.npz）"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="配置文件路径（可选）"
    )
    
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="不保存调试可视化图像"
    )
    
    args = parser.parse_args()
    
    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return 1
    
    prepare_training_data(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        config_path=args.config,
        save_debug=not args.no_debug
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
