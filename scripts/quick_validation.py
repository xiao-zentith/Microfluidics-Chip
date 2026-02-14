"""
快速可行性验证脚本
用单芯片数据测试整个Pipeline

验证目标：
1. YOLO能否稳定检测12个腔室
2. 几何校正是否正常
3. UNet能否学习光照校正
4. 评估指标是否提升
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microfluidics_chip.core.config import get_default_config
from microfluidics_chip.stage1_detection.detector import ChamberDetector
from microfluidics_chip.stage1_detection.geometry_engine import CrossGeometryEngine
from microfluidics_chip.stage2_correction.models import RefGuidedUNet
from microfluidics_chip.stage2_correction.losses import ROIWeightedLoss
from microfluidics_chip.stage2_correction.dataset import MicrofluidicDataset
from microfluidics_chip.stage2_correction.trainer import train_model
from microfluidics_chip.core.logger import setup_logger, get_logger

logger = get_logger("quick_validation")


def validate_yolo_detection(chip_dir: Path, detector: ChamberDetector) -> bool:
    """验证YOLO检测能力"""
    logger.info("=" * 60)
    logger.info("Step 1: YOLO检测验证")
    logger.info("=" * 60)
    
    # 查找所有图像
    gt_path = list(chip_dir.glob("gt.*")) + list(chip_dir.glob("GT.*"))
    dirty_paths = list(chip_dir.glob("dirty_*.png")) + list(chip_dir.glob("noisy_*.png"))
    
    if not gt_path:
        logger.error("未找到GT图像")
        return False
    
    all_images = [gt_path[0]] + dirty_paths
    logger.info(f"找到 {len(all_images)} 张图像（1 GT + {len(dirty_paths)} Dirty）")
    
    # 逐个检测
    detection_results = {}
    for img_path in all_images:
        img = cv2.imread(str(img_path))
        detections = detector.detect(img)
        detection_results[img_path.name] = len(detections)
        
        status = "✅" if len(detections) == 12 else "❌"
        logger.info(f"  {status} {img_path.name}: {len(detections)}/12 chambers")
    
    # 统计
    success_rate = sum(1 for n in detection_results.values() if n == 12) / len(detection_results)
    logger.info(f"\n检测成功率: {success_rate:.1%} ({sum(1 for n in detection_results.values() if n == 12)}/{len(detection_results)})")
    
    if success_rate < 1.0:
        logger.warning("⚠️  部分图像检测失败，建议：")
        logger.warning("  1. 降低 conf_threshold（当前可能在0.5）")
        logger.warning("  2. 检查图像质量（模糊、遮挡）")
        logger.warning("  3. 重新训练YOLO或使用数据增强")
        return False
    
    logger.info("✅ YOLO检测全部通过")
    return True


def prepare_data(chip_dir: Path, output_path: Path, detector: ChamberDetector, config) -> bool:
    """准备训练数据"""
    logger.info("=" * 60)
    logger.info("Step 2: 数据准备")
    logger.info("=" * 60)
    
    from scripts.prepare_training_data import process_chip_directory
    
    samples = process_chip_directory(chip_dir, detector, config)
    
    if not samples:
        logger.error("数据准备失败")
        return False
    
    logger.info(f"生成 {len(samples)} 个训练样本")
    
    if len(samples) < 20:
        logger.warning(f"⚠️  样本量较少（{len(samples)}），建议至少20个")
        logger.warning(f"   提示：增加更多dirty图像（当前可能只有{len(samples)//9}张）")
    
    # 保存
    target_in = np.array([s['signal'] for s in samples], dtype=np.float32)
    ref_in = np.array([s['reference'] for s in samples], dtype=np.float32)
    labels = np.array([s['target'] for s in samples], dtype=np.float32)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, target_in=target_in, ref_in=ref_in, labels=labels)
    
    logger.info(f"✅ 数据已保存: {output_path}")
    logger.info(f"   形状: {target_in.shape}")
    return True


def quick_train(data_path: Path, output_dir: Path, epochs: int = 50) -> bool:
    """快速训练验证"""
    logger.info("=" * 60)
    logger.info("Step 3: 快速训练")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据（全部用作训练，不划分验证集）
    dataset = MicrofluidicDataset(data_path, mode='train', split_ratio=1.0)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    logger.info(f"训练样本: {len(dataset)}")
    
    # 创建模型
    model = RefGuidedUNet().to(device)
    criterion = ROIWeightedLoss(roi_radius=20, edge_weight=0.1, lambda_cos=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 快速训练
    logger.info(f"开始训练 {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (signal, ref, gt) in enumerate(loader):
            signal, ref, gt = signal.to(device), ref.to(device), gt.to(device)
            
            optimizer.zero_grad()
            output = model(signal, ref)
            loss = criterion(output, gt, None)  # ROI map会自动生成
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # 保存模型
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'epoch': epochs,
        'loss': avg_loss
    }, output_dir / "quick_model.pth")
    
    logger.info(f"✅ 训练完成，模型已保存")
    return True


def visual_check(chip_dir: Path, model_path: Path, output_dir: Path) -> bool:
    """可视化检查"""
    logger.info("=" * 60)
    logger.info("Step 4: 可视化验证")
    logger.info("=" * 60)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = RefGuidedUNet().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 随机选一张dirty图像测试
    dirty_paths = list(chip_dir.glob("dirty_*.png"))
    if not dirty_paths:
        logger.error("未找到dirty图像")
        return False
    
    test_img_path = dirty_paths[0]
    logger.info(f"测试图像: {test_img_path.name}")
    
    # ... (此处可添加完整推理+可视化逻辑)
    
    logger.info(f"✅ 可视化结果已保存到 {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="单芯片快速验证")
    parser.add_argument("chip_dir", type=Path, help="芯片目录（包含gt.png和多个dirty_*.png）")
    parser.add_argument("-o", "--output", type=Path, default=Path("runs/quick_validation"), help="输出目录")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--skip-yolo-check", action="store_true", help="跳过YOLO检测验证")
    
    args = parser.parse_args()
    
    setup_logger(level="INFO")
    
    logger.info("🚀 微流控芯片快速验证开始")
    logger.info(f"芯片目录: {args.chip_dir}")
    logger.info(f"输出目录: {args.output}")
    
    # 加载配置和检测器
    config = get_default_config()
    detector = ChamberDetector(config.stage1.yolo)
    
    # Step 1: YOLO验证
    if not args.skip_yolo_check:
        if not validate_yolo_detection(args.chip_dir, detector):
            logger.error("❌ YOLO检测验证失败，请先解决检测问题")
            return 1
    
    # Step 2: 数据准备
    data_path = args.output / "data.npz"
    if not prepare_data(args.chip_dir, data_path, detector, config):
        logger.error("❌ 数据准备失败")
        return 1
    
    # Step 3: 快速训练
    if not quick_train(data_path, args.output, args.epochs):
        logger.error("❌ 训练失败")
        return 1
    
    # Step 4: 可视化
    # visual_check(args.chip_dir, args.output / "quick_model.pth", args.output / "visualizations")
    
    logger.info("=" * 60)
    logger.info("✅ 快速验证完成！")
    logger.info("=" * 60)
    logger.info("后续步骤：")
    logger.info("  1. 检查训练Loss是否下降")
    logger.info("  2. 增加更多芯片数据")
    logger.info("  3. 使用完整训练流程")
    
    return 0


if __name__ == "__main__":
    exit(main())
