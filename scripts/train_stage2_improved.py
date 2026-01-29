"""
Stage2 UNet 改进训练脚本

改进点：
1. 支持数据增强（训练集）
2. 优化的损失函数参数
3. 改进的学习率调度器
4. 更长的训练周期

用法：
    python scripts/train_stage2_improved.py data/training.npz -o runs/improved -e 100
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import typer
from rich.console import Console

from microfluidics_chip.stage2_correction.models.dual_stream_unet import RefGuidedUNet
from microfluidics_chip.stage2_correction.losses import ROIWeightedLoss
from microfluidics_chip.stage2_correction.dataset import MicrofluidicDataset
from microfluidics_chip.stage2_correction.trainer import train_model
from microfluidics_chip.core.logger import setup_logger, get_logger

app = typer.Typer()
console = Console()
logger = get_logger("scripts.train_stage2_improved")


@app.command()
def main(
    npz_path: Path = typer.Argument(..., help="训练数据npz文件路径"),
    output_dir: Path = typer.Option("runs/improved_training", "--output", "-o", help="训练输出目录"),
    epochs: int = typer.Option(100, "--epochs", "-e", help="训练轮数"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="批次大小"),
    lr: float = typer.Option(3e-4, "--lr", help="学习率"),
    device: str = typer.Option("cuda", "--device", "-d", help="设备 (cuda/cpu)"),
    
    # 损失函数参数（优化的默认值）
    roi_radius: int = typer.Option(25, "--roi-radius", help="ROI半径"),
    edge_weight: float = typer.Option(0.3, "--edge-weight", help="边缘权重"),
    lambda_cos: float = typer.Option(0.5, "--lambda-cos", help="余弦损失权重"),
    
    # 数据增强
    augment: bool = typer.Option(True, "--augment/--no-augment", help="启用训练集数据增强"),
    aug_intensity: float = typer.Option(0.3, "--aug-intensity", help="光照增强强度 [0-1], 0.3=温和, 0.5=中等, 0.7=激进"),
    
    # 其他
    num_workers: int = typer.Option(4, "--workers", "-w", help="数据加载器线程数"),
    visualize_every: int = typer.Option(10, "--visualize-every", help="可视化间隔"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    改进的Stage2 UNet训练
    
    目标：PSNR > 30 dB
    
    示例：
        python scripts/train_stage2_improved.py data/training.npz -o runs/improved -e 100
    """
    # 设置日志
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(
        level=log_level,
        log_file=output_dir / "training.log"
    )
    
    console.print("[bold green]Starting Improved Stage2 UNet Training[/bold green]")
    console.print(f"NPZ Path: {npz_path}")
    console.print(f"Output Dir: {output_dir}")
    console.print(f"Device: {device}")
    
    if augment:
        console.print(f"Data Augmentation: [green]ENABLED[/green] (intensity={aug_intensity})")
        console.print(f"  - Geometric: Flip + Rotate")
        console.print(f"  - Lighting: Gamma + Gradient + Shadow (微流控专用)")
    else:
        console.print(f"Data Augmentation: [red]DISABLED[/red]")
    
    # 检查数据文件
    if not npz_path.exists():
        console.print(f"[bold red]Error: NPZ file not found: {npz_path}[/bold red]")
        raise typer.Exit(code=1)
    
    # 设置设备
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]Warning: CUDA not available, using CPU[/yellow]")
    
    # ==================== 加载数据 ====================
    console.print("[bold]Loading datasets...[/bold]")
    
    train_dataset = MicrofluidicDataset(
        npz_path, 
        mode='train', 
        split_ratio=0.9, 
        augment=augment,  # 训练集增强
        aug_intensity=aug_intensity
    )
    val_dataset = MicrofluidicDataset(
        npz_path, 
        mode='val', 
        split_ratio=0.9,
        augment=False  # 验证集不增强
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=(device == "cuda")
    )
    
    console.print(f"Train samples: {len(train_dataset)}")
    console.print(f"Val samples: {len(val_dataset)}")
    
    # ==================== 初始化模型 ====================
    console.print("[bold]Initializing model...[/bold]")
    
    model = RefGuidedUNet(
        in_channels=3,
        out_channels=3,
        features=[64, 128, 256, 512]
    ).to(device_obj)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Total parameters: {total_params:,}")
    console.print(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== 损失函数 ====================
    criterion = ROIWeightedLoss(
        roi_radius=roi_radius,
        edge_weight=edge_weight,
        lambda_cos=lambda_cos
    ).to(device_obj)
    
    console.print(f"Loss: ROI-Weighted (radius={roi_radius}, edge={edge_weight}, λ_cos={lambda_cos})")
    
    # ==================== 训练 ====================
    console.print("[bold]Starting training...[/bold]")
    console.print(f"[yellow]Target: PSNR > 30 dB[/yellow]")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device_obj,
        epochs=epochs,
        lr=lr,
        save_dir=output_dir,
        visualize_every=visualize_every
    )
    
    # ==================== 汇总 ====================
    console.print("\n[bold green]Training Complete![/bold green]")
    console.print(f"Best PSNR: {history['best_psnr']:.2f} dB (Epoch {history['best_epoch']})")
    console.print(f"Model saved to: {output_dir / 'best_model.pth'}")
    
    # 目标检查
    if history['best_psnr'] >= 30.0:
        console.print("[bold green]✓ Target achieved: PSNR >= 30 dB[/bold green]")
    elif history['best_psnr'] >= 25.0:
        console.print("[bold yellow]⚠ Close to target: 25 <= PSNR < 30 dB[/bold yellow]")
        console.print("[yellow]Suggestions: 1) Add more data  2) Train longer  3) Adjust hyperparameters[/yellow]")
    else:
        console.print("[bold red]✗ Target not met: PSNR < 25 dB[/bold red]")
        console.print("[red]Suggestions: 1) Check data quality  2) Increase data augmentation  3) Try different loss weights[/red]")


if __name__ == "__main__":
    app()
