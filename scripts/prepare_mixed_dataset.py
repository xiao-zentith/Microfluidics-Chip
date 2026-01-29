"""
混合数据集生成脚本

结合两种数据生成方式：
1. 真实数据：1GT + 多Dirty图 → 高质量、多样性
2. 合成数据：1GT × 倍率 → 数量补充、可控性

使用场景：
- 真实数据不足时，用合成数据补充
- 平衡数据分布
- 增强训练效果

使用方法：
  python scripts/prepare_mixed_dataset.py \
    --real dataset/real_training \
    --synthetic dataset/clean_images \
    --output processed_data/mixed_training.npz \
    --synthetic-multiplier 50
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microfluidics_chip.core.config import get_default_config, load_config_from_yaml
from microfluidics_chip.stage1_detection.detector import ChamberDetector
from microfluidics_chip.stage1_detection.geometry_engine import CrossGeometryEngine
from microfluidics_chip.stage1_detection.synthesizer import FullChipSynthesizer
from microfluidics_chip.core.logger import get_logger, setup_logger

logger = get_logger("prepare_mixed_dataset")


def load_real_data(real_dir: Path, detector, config) -> tuple:
    """
    加载真实数据（1GT + 多Dirty）
    
    返回: (signals, references, targets)
    """
    from prepare_training_data import process_chip_directory
    
    chip_dirs = [d for d in real_dir.iterdir() if d.is_dir()]
    logger.info(f"Processing {len(chip_dirs)} real chip directories...")
    
    all_samples = []
    for chip_dir in tqdm(chip_dirs, desc="Real data"):
        samples = process_chip_directory(chip_dir, detector, config)
        all_samples.extend(samples)
    
    if not all_samples:
        return None, None, None
    
    signals = np.array([s['signal'] for s in all_samples], dtype=np.float32)
    references = np.array([s['reference'] for s in all_samples], dtype=np.float32)
    targets = np.array([s['target'] for s in all_samples], dtype=np.float32)
    
    logger.info(f"✓ Real data: {len(all_samples)} samples")
    return signals, references, targets


def generate_synthetic_data(
    synthetic_dir: Path,
    detector: ChamberDetector,
    config,
    multiplier: int = 50
) -> tuple:
    """
    生成合成数据（1GT × 倍率）
    
    返回: (signals, references, targets)
    """
    # 初始化合成器
    synthesizer = FullChipSynthesizer(
        detector=detector,
        geometry_config=config.stage1.geometry,
        class_id_blank=config.stage1.geometry.class_id_blank
    )
    
    # 收集GT图像
    gt_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        gt_files.extend(synthetic_dir.glob(f"*{ext}"))
    
    logger.info(f"Processing {len(gt_files)} GT images for synthesis (multiplier={multiplier})...")
    
    all_signals, all_refs, all_targets = [], [], []
    
    for gt_file in tqdm(gt_files, desc="Synthetic data"):
        triplets = []
        for _ in range(multiplier):
            batch = synthesizer.generate_triplets(gt_file)
            triplets.extend(batch)
        
        # 提取数据
        for t in triplets:
            all_signals.append(t['target_in'])
            all_refs.append(t['ref_in'])
            all_targets.append(t['target_gt'])
    
    if not all_signals:
        return None, None, None
    
    signals = np.array(all_signals, dtype=np.float32)
    references = np.array(all_refs, dtype=np.float32)
    targets = np.array(all_targets, dtype=np.float32)
    
    # 调整references形状以匹配真实数据格式
    # synthesizer返回的是单个ref，需要扩展维度
    if len(references.shape) == 4:  # (N, H, W, 3)
        references = references[:, np.newaxis, :, :, :]  # (N, 1, H, W, 3)
    
    logger.info(f"✓ Synthetic data: {len(all_signals)} samples")
    return signals, references, targets


def prepare_mixed_dataset(
    real_dir: Path = None,
    synthetic_dir: Path = None,
    output_path: Path = None,
    config_path: Path = None,
    synthetic_multiplier: int = 50,
    real_weight: float = 1.0,
    synthetic_weight: float = 1.0
):
    """
    准备混合数据集
    
    :param real_dir: 真实数据目录（1GT + 多Dirty结构）
    :param synthetic_dir: 合成数据GT图像目录
    :param output_path: 输出NPZ路径
    :param config_path: 配置文件
    :param synthetic_multiplier: 合成数据倍率
    :param real_weight: 真实数据权重（用于采样平衡）
    :param synthetic_weight: 合成数据权重
    """
    setup_logger(level="INFO")
    
    # 加载配置
    if config_path and config_path.exists():
        config = load_config_from_yaml(config_path)
    else:
        config = get_default_config()
    
    # 初始化检测器
    logger.info("Initializing YOLO detector...")
    detector = ChamberDetector(config.stage1.yolo)
    logger.info("✓ Detector initialized")
    
    # 加载/生成数据
    signals_list, refs_list, targets_list = [], [], []
    
    if real_dir and real_dir.exists():
        logger.info(f"\n{'='*60}")
        logger.info("Processing Real Data (1GT + Multi-Dirty)")
        logger.info(f"{'='*60}")
        
        s, r, t = load_real_data(real_dir, detector, config)
        if s is not None:
            signals_list.append(s)
            refs_list.append(r)
            targets_list.append(t)
            logger.info(f"Real data shape: {s.shape}")
    
    if synthetic_dir and synthetic_dir.exists():
        logger.info(f"\n{'='*60}")
        logger.info("Generating Synthetic Data (1GT × Multiplier)")
        logger.info(f"{'='*60}")
        
        s, r, t = generate_synthetic_data(
            synthetic_dir, detector, config, synthetic_multiplier
        )
        if s is not None:
            signals_list.append(s)
            refs_list.append(r)
            targets_list.append(t)
            logger.info(f"Synthetic data shape: {s.shape}")
    
    if not signals_list:
        logger.error("No data generated!")
        return
    
    # 合并数据
    logger.info(f"\n{'='*60}")
    logger.info("Merging Datasets")
    logger.info(f"{'='*60}")
    
    signals = np.concatenate(signals_list, axis=0)
    references = np.concatenate(refs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    logger.info(f"Final dataset:")
    logger.info(f"  - Signals:    {signals.shape}")
    logger.info(f"  - References: {references.shape}")
    logger.info(f"  - Targets:    {targets.shape}")
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        signals=signals,
        references=references,
        targets=targets
    )
    
    logger.info(f"\n✓ Mixed dataset saved to: {output_path}")
    logger.info(f"Total samples: {len(signals)}")
    
    # 统计
    logger.info(f"\n{'='*60}")
    logger.info("Dataset Composition:")
    if len(signals_list) == 2:
        logger.info(f"  - Real data:      {len(signals_list[0])} samples ({len(signals_list[0])/len(signals)*100:.1f}%)")
        logger.info(f"  - Synthetic data: {len(signals_list[1])} samples ({len(signals_list[1])/len(signals)*100:.1f}%)")
    logger.info(f"  - Total:          {len(signals)} samples")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mixed training dataset (Real + Synthetic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--real",
        type=Path,
        default=None,
        help="真实数据目录（1GT + 多Dirty结构）"
    )
    
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=None,
        help="合成数据GT图像目录"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="输出NPZ文件路径"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--synthetic-multiplier",
        type=int,
        default=50,
        help="合成数据倍率（默认：50）"
    )
    
    args = parser.parse_args()
    
    if not args.real and not args.synthetic:
        print("Error: Must specify at least one of --real or --synthetic")
        return 1
    
    prepare_mixed_dataset(
        real_dir=args.real,
        synthetic_dir=args.synthetic,
        output_path=args.output,
        config_path=args.config,
        synthetic_multiplier=args.synthetic_multiplier
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
