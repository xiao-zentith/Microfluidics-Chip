"""
UNet 训练器
继承自 v1.0 的 unet/model/train.py

功能：
- 训练循环（train_one_epoch）
- 验证（evaluate）
- 可视化（visualize_results）
- 指标计算（calculate_psnr）
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random
from typing import Tuple, Optional
from ..core.logger import get_logger

logger = get_logger("stage2_correction.trainer")


# ==================== 工具函数 ====================

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    计算 PSNR (dB)
    
    :param img1: 预测图像 (B, 3, H, W)
    :param img2: 目标图像 (B, 3, H, W)
    :return: PSNR 值（dB）
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ==================== 训练引擎 ====================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float, float]:
    """
    训练一个 epoch
    
    :param model: 模型
    :param loader: 数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :param epoch: 当前 epoch
    :return: (avg_total_loss, avg_pixel_loss, avg_cos_loss)
    """
    model.train()
    running_loss = 0.0
    running_pix = 0.0
    running_cos = 0.0
    
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for signal, ref, gt in loop:
        signal, ref, gt = signal.to(device), ref.to(device), gt.to(device)
        
        optimizer.zero_grad()
        output = model(signal, ref)
        
        # 计算混合损失
        loss, l_pix, l_cos = criterion(output, gt)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_pix += l_pix.item()
        running_cos += l_cos.item()
        
        loop.set_postfix(loss=f"{loss.item():.4f}", pix=f"{l_pix.item():.4f}")
    
    count = len(loader)
    return running_loss / count, running_pix / count, running_cos / count


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    验证模型
    
    :param model: 模型
    :param loader: 验证数据加载器
    :param criterion: 损失函数
    :param device: 设备
    :return: (avg_loss, avg_psnr)
    """
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    
    with torch.no_grad():
        for signal, ref, gt in loader:
            signal, ref, gt = signal.to(device), ref.to(device), gt.to(device)
            output = model(signal, ref)
            loss, _, _ = criterion(output, gt)
            
            val_loss += loss.item()
            val_psnr += calculate_psnr(output, gt).item()
    
    return val_loss / len(loader), val_psnr / len(loader)


# ==================== 结果可视化 ====================

def visualize_results(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    epoch: int,
    save_dir: Path
):
    """
    结果可视化（论文级展示）
    
    展示：
    1. Input (Dirty)
    2. Ours Output
    3. Ground Truth
    4. Error Before
    5. Error After
    
    :param model: 模型
    :param dataset: 数据集
    :param device: 设备
    :param epoch: 当前 epoch
    :param save_dir: 保存目录
    """
    model.eval()
    
    # 随机取样
    idx = random.randint(0, len(dataset) - 1)
    signal, ref, gt = dataset[idx]
    
    # 推理
    signal_in = signal.unsqueeze(0).to(device)
    ref_in = ref.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(signal_in, ref_in)
    
    # 辅助函数: Tensor -> Numpy Image (H, W, 3)
    def to_img(t: torch.Tensor) -> np.ndarray:
        return t.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    img_in = to_img(signal_in)
    img_out = to_img(output)
    img_gt = to_img(gt)
    
    # 计算差异图（RGB 距离）
    diff_before = np.sqrt(np.sum((img_in - img_gt)**2, axis=2))
    diff_after = np.sqrt(np.sum((img_out - img_gt)**2, axis=2))
    
    # 绘图
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(img_in)
    axes[0].set_title("Input (Dirty)")
    axes[0].axis('off')
    
    axes[1].imshow(img_out)
    axes[1].set_title("Ours Output")
    axes[1].axis('off')
    
    axes[2].imshow(img_gt)
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    # 差异热力图
    axes[3].imshow(diff_before, cmap='jet', vmin=0, vmax=0.5)
    axes[3].set_title("Error Before")
    axes[3].axis('off')
    
    im = axes[4].imshow(diff_after, cmap='jet', vmin=0, vmax=0.5)
    axes[4].set_title("Error After (Ours)")
    axes[4].axis('off')
    
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    
    save_path = save_dir / f"epoch_{epoch}_vis.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved: {save_path}")


# ==================== 训练曲线可视化 ====================

def plot_training_curves(history: dict, save_dir: Path):
    """
    绘制训练曲线
    
    展示：
    - Loss曲线（Train vs Val）
    - PSNR曲线（Val）
    - Learning Rate变化
    
    :param history: 训练历史字典
    :param save_dir: 保存目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Training Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Validation PSNR
    axes[0, 1].plot(epochs, history['val_psnr'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0, 1].set_title('Validation PSNR', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 标记最佳点
    best_idx = np.argmax(history['val_psnr'])
    best_psnr = history['val_psnr'][best_idx]
    axes[0, 1].scatter([best_idx + 1], [best_psnr], color='red', s=100, zorder=5)
    axes[0, 1].annotate(
        f'Best: {best_psnr:.2f} dB\n(Epoch {best_idx + 1})',
        xy=(best_idx + 1, best_psnr),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
    
    # 3. Loss Components (Train)
    axes[1, 0].plot(epochs, history['train_pixel_loss'], 'b-', label='Pixel Loss', linewidth=2)
    axes[1, 0].plot(epochs, history['train_cos_loss'], 'r-', label='Cosine Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Training Loss Components', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Learning Rate
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No LR history', 
                        ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved: {save_path}")


# ==================== 完整训练流程 ====================

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-4,
    save_dir: Path = Path("runs/stage2_training"),
    visualize_every: int = 5
) -> dict:
    """
    完整训练流程
    
    :param model: 模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param criterion: 损失函数
    :param device: 设备
    :param epochs: 训练轮数
    :param lr: 学习率
    :param save_dir: 保存目录
    :param visualize_every: 每隔多少 epoch 可视化一次
    :return: 训练历史（best_psnr, best_epoch等）
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_pixel_loss': [],
        'train_cos_loss': [],
        'val_loss': [],
        'val_psnr': [],
        'learning_rate': []
    }
    
    best_psnr = 0.0
    best_epoch = 0
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Saving to: {save_dir}")
    
    for epoch in range(1, epochs + 1):
        # Train
        t_loss, t_pix, t_cos = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        v_loss, v_psnr = evaluate(model, val_loader, criterion, device)
        
        # 记录历史
        history['train_loss'].append(t_loss)
        history['train_pixel_loss'].append(t_pix)
        history['train_cos_loss'].append(t_cos)
        history['val_loss'].append(v_loss)
        history['val_psnr'].append(v_psnr)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Log
        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {t_loss:.4f} | "
            f"Val Loss: {v_loss:.4f} | "
            f"Val PSNR: {v_psnr:.2f} dB | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        # Save Best
        if v_psnr > best_psnr:
            best_psnr = v_psnr
            best_epoch = epoch
            model_path = save_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"*** Best Model Saved (PSNR: {best_psnr:.2f} dB) ***")
        
        # Visualize
        if epoch % visualize_every == 0:
            visualize_results(
                model, val_loader.dataset, device, epoch, save_dir
            )
        
        # Update LR (monitor PSNR)
        scheduler.step(v_psnr)
    
    # 训练结束后绘制曲线
    logger.info("=" * 60)
    logger.info("Training complete! Generating training curves...")
    plot_training_curves(history, save_dir)
    
    logger.info(f"Best PSNR: {best_psnr:.2f} dB at epoch {best_epoch}")
    logger.info(f"All results saved to: {save_dir}")
    logger.info("=" * 60)
    
    return {
        'best_psnr': best_psnr,
        'best_epoch': best_epoch,
        'final_epoch': epochs,
        'history': history
    }

