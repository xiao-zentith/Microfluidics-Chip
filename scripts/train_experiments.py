"""
统一实验训练脚本

用法：
    python scripts/train_experiments.py \
        --config configs/experiments/ablation_a_dual.yaml \
        --output-dir runs/experiments

功能：
- 加载实验配置
- 根据 model_type 实例化对应模型
- 统一的训练循环
- 保存实验 manifest（配置快照 + Git hash）
- 输出训练曲线和最佳模型
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# 项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microfluidics_chip.stage2_correction.models import RefGuidedUNet, SingleStreamUNet
from microfluidics_chip.stage2_correction.losses import ROIWeightedLoss
from microfluidics_chip.stage2_correction.dataset import MicrofluidicDataset
from microfluidics_chip.stage2_correction.trainer import train_one_epoch, evaluate, visualize_results, plot_training_curves
from microfluidics_chip.stage2_correction.evaluation import evaluate_tensor_batch
from microfluidics_chip.core.logger import get_logger

logger = get_logger("scripts.train_experiments")


def get_git_hash() -> str:
    """获取当前 Git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def load_config(config_path: Path) -> Dict[str, Any]:
    """加载实验配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """根据配置创建模型"""
    model_config = config["stage2"]["model"]
    model_type = model_config.get("model_type", "dual_stream")
    
    features = model_config.get("features", [64, 128, 256, 512])
    in_channels = model_config.get("in_channels", 3)
    out_channels = model_config.get("out_channels", 3)
    
    if model_type == "dual_stream":
        logger.info("Creating RefGuidedUNet (dual-stream)")
        return RefGuidedUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features
        )
    elif model_type == "single_stream":
        logger.info("Creating SingleStreamUNet (single-stream baseline)")
        return SingleStreamUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_loss(config: Dict[str, Any]) -> torch.nn.Module:
    """根据配置创建损失函数"""
    loss_config = config["stage2"].get("loss", {})
    
    return ROIWeightedLoss(
        roi_radius=loss_config.get("roi_radius", 20),
        edge_weight=loss_config.get("edge_weight", 0.1),
        lambda_cos=loss_config.get("lambda_cos", 0.2)
    )


def save_manifest(output_dir: Path, config: Dict[str, Any], args: argparse.Namespace):
    """保存实验 manifest"""
    manifest = {
        "experiment_name": config.get("experiment_name", "unnamed"),
        "experiment_type": config.get("experiment_type", "unknown"),
        "description": config.get("description", ""),
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "config_path": str(args.config),
        "config": config,
        "command_line": " ".join(sys.argv)
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Experiment Training Script")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("runs/experiments"),
        help="Output directory for experiment results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from config"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    logger.info(f"Loaded config: {config.get('experiment_name', 'unnamed')}")
    
    # 创建输出目录
    experiment_name = config.get("experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # 保存配置副本
    shutil.copy(args.config, run_dir / "config.yaml")
    
    # 保存 manifest
    save_manifest(run_dir, config, args)
    
    # 设置设备
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # 加载数据
    training_config = config.get("training", {})
    data_path = Path(training_config.get("data_path", "data/training.npz"))
    
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        logger.info("Please run data preparation first: python scripts/prepare_training_data.py")
        return 1
    
    logger.info(f"Loading data from {data_path}")
    dataset = MicrofluidicDataset(data_path)
    
    # 划分训练/验证集
    val_split = training_config.get("val_split", 0.2)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    batch_size = training_config.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Dataset: {len(dataset)} samples (train={train_size}, val={val_size})")
    
    # 创建模型
    model = create_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # 创建损失函数
    criterion = create_loss(config).to(device)
    
    # 创建优化器
    lr = training_config.get("learning_rate", 1e-4)
    weight_decay = training_config.get("weight_decay", 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    epochs = args.epochs or training_config.get("epochs", 100)
    scheduler_type = training_config.get("scheduler", "cosine")
    
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None
    
    # 训练历史
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "lr": []
    }
    
    best_psnr = 0.0
    
    # 恢复训练
    start_epoch = 0
    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get("best_psnr", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # 训练循环
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(start_epoch, epochs):
        # 训练
        train_loss, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_psnr = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["lr"].append(current_lr)
        
        # 日志
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PSNR: {val_psnr:.2f} dB | "
            f"LR: {current_lr:.2e}"
        )
        
        # 保存最佳模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_psnr": best_psnr,
                "config": config
            }, weights_dir / "best_model.pth")
            logger.info(f"New best model saved (PSNR: {best_psnr:.2f} dB)")
        
        # 定期可视化
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            visualize_results(model, dataset, device, epoch, vis_dir)
        
        # 定期保存 checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_psnr": best_psnr,
                "history": history
            }, weights_dir / f"checkpoint_epoch{epoch+1}.pth")
    
    # 保存最终模型
    torch.save({
        "epoch": epochs - 1,
        "model": model.state_dict(),
        "best_psnr": best_psnr,
        "config": config
    }, weights_dir / "final_model.pth")
    
    # 绘制训练曲线
    plot_training_curves(history, run_dir)
    
    # 保存训练历史
    with open(run_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # 保存最终指标
    final_metrics = {
        "best_psnr": best_psnr,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "total_epochs": epochs,
        "model_params": num_params
    }
    
    with open(run_dir / "metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info(f"Training complete! Best PSNR: {best_psnr:.2f} dB")
    logger.info(f"Results saved to: {run_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
