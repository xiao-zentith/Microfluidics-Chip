"""
统一实验评估脚本

用法：
    python scripts/evaluate_experiments.py \
        --experiment-dirs runs/ablation_a_dual runs/ablation_a_single \
        --test-data data/test_set.npz \
        --output results/ablation_a_comparison.json

功能：
- 加载多个实验的训练结果
- 在相同测试集上评估
- 生成对比表格（Markdown/CSV）
- 生成可视化图表（箱线图、误差分布）
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microfluidics_chip.stage2_correction.models import RefGuidedUNet, SingleStreamUNet
from microfluidics_chip.stage2_correction.dataset import MicrofluidicDataset
from microfluidics_chip.stage2_correction.evaluation import (
    evaluate_tensor_batch, EvaluationMetrics
)
from microfluidics_chip.core.logger import get_logger

logger = get_logger("scripts.evaluate_experiments")


def load_model_from_experiment(experiment_dir: Path, device: torch.device) -> torch.nn.Module:
    """从实验目录加载模型"""
    weights_path = experiment_dir / "weights" / "best_model.pth"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location=device)
    config = checkpoint.get("config", {})
    
    # 获取模型类型
    model_config = config.get("stage2", {}).get("model", {})
    model_type = model_config.get("model_type", "dual_stream")
    features = model_config.get("features", [64, 128, 256, 512])
    
    # 创建模型
    if model_type == "dual_stream":
        model = RefGuidedUNet(features=features)
    elif model_type == "single_stream":
        model = SingleStreamUNet(features=features)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_type} model from {experiment_dir.name}")
    
    return model, config


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    roi_radius: int = 20
) -> Dict[str, float]:
    """评估单个模型"""
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # MicrofluidicDataset 返回元组: (signal, ref, gt)
            targets_in, refs_in, labels = batch
            targets_in = targets_in.to(device)
            refs_in = refs_in.to(device)
            labels = labels.to(device)
            
            # 前向传播
            preds = model(targets_in, refs_in)
            
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
    
    # 拼接所有批次
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算指标
    metrics = evaluate_tensor_batch(all_preds, all_targets, roi_radius)
    
    return metrics.to_dict()


def generate_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """生成Markdown对比表格"""
    # 获取所有指标
    first_result = list(results.values())[0]
    metrics = list(first_result.keys())
    
    # 表头
    header = "| Experiment | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join([":---:"] * (len(metrics) + 1)) + "|"
    
    # 表内容
    rows = []
    for exp_name, exp_metrics in results.items():
        values = [f"{exp_metrics[m]:.4f}" for m in metrics]
        rows.append(f"| {exp_name} | " + " | ".join(values) + " |")
    
    return "\n".join([header, separator] + rows)


def generate_comparison_plot(
    results: Dict[str, Dict[str, float]],
    output_path: Path
):
    """生成对比图表"""
    experiments = list(results.keys())
    metrics = ["psnr", "ssim", "rmse", "cosine_sim"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Experiment Comparison", fontsize=14, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results[exp][metric] for exp in experiments]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
        bars = ax.bar(experiments, values, color=colors)
        
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9
            )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plot to {output_path}")


def generate_roi_comparison_plot(
    results: Dict[str, Dict[str, float]],
    output_path: Path
):
    """生成ROI vs Edge RMSE对比图"""
    experiments = list(results.keys())
    
    roi_rmse = [results[exp]["roi_rmse"] for exp in experiments]
    edge_rmse = [results[exp]["edge_rmse"] for exp in experiments]
    
    x = np.arange(len(experiments))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, roi_rmse, width, label='ROI RMSE', color='#2ecc71')
    bars2 = ax.bar(x + width/2, edge_rmse, width, label='Edge RMSE', color='#e74c3c')
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('RMSE')
    ax.set_title('ROI vs Edge RMSE Comparison\n(Lower ROI RMSE indicates effective ROI weighting)')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend()
    
    # 添加数值标签
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROI comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Experiment Evaluation Script")
    parser.add_argument(
        "--experiment-dirs", "-e",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to experiment directories"
    )
    parser.add_argument(
        "--test-data", "-t",
        type=Path,
        required=True,
        help="Path to test data (npz file)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/comparison.json"),
        help="Output path for comparison results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--roi-radius",
        type=int,
        default=20,
        help="ROI radius for evaluation"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # 加载测试数据
    if not args.test_data.exists():
        logger.error(f"Test data not found: {args.test_data}")
        return 1
    
    test_dataset = MicrofluidicDataset(args.test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    logger.info(f"Loaded test data: {len(test_dataset)} samples")
    
    # 评估每个实验
    results = {}
    
    for exp_dir in args.experiment_dirs:
        if not exp_dir.exists():
            logger.warning(f"Experiment directory not found: {exp_dir}")
            continue
        
        try:
            # 加载模型
            model, config = load_model_from_experiment(exp_dir, device)
            
            # 评估
            exp_name = config.get("experiment_name", exp_dir.name)
            logger.info(f"Evaluating: {exp_name}")
            
            metrics = evaluate_model(model, test_loader, device, args.roi_radius)
            results[exp_name] = metrics
            
            logger.info(f"  PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {exp_dir}: {e}")
            continue
    
    if not results:
        logger.error("No experiments were successfully evaluated")
        return 1
    
    # 生成对比表格
    table = generate_comparison_table(results)
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(table)
    print("=" * 60 + "\n")
    
    # 保存结果
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_data": str(args.test_data),
        "num_test_samples": len(test_dataset),
        "experiments": results,
        "comparison_table": table
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to {args.output}")
    
    # 保存Markdown表格
    md_path = args.output.with_suffix('.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Experiment Comparison Results\n\n")
        f.write(f"**Test Data**: {args.test_data}\n\n")
        f.write(f"**Samples**: {len(test_dataset)}\n\n")
        f.write(f"**Timestamp**: {datetime.now().isoformat()}\n\n")
        f.write("## Metrics Comparison\n\n")
        f.write(table)
        f.write("\n")
    
    logger.info(f"Saved Markdown to {md_path}")
    
    # 生成图表
    plot_path = args.output.with_name(args.output.stem + "_comparison.png")
    generate_comparison_plot(results, plot_path)
    
    roi_plot_path = args.output.with_name(args.output.stem + "_roi_comparison.png")
    generate_roi_comparison_plot(results, roi_plot_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
