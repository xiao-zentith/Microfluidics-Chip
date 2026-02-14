"""
评估指标模块

提供图像质量评估的标准指标：
- PSNR: 峰值信噪比
- SSIM: 结构相似性
- RMSE: 均方根误差
- Cosine Similarity: 光谱保真度
- ROI分区评估: 核心区vs边缘区

所有指标的设计理由见 implementation_plan.md 第五部分
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    psnr: float
    ssim: float
    rmse: float
    cosine_sim: float
    roi_rmse: float
    edge_rmse: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "rmse": self.rmse,
            "cosine_sim": self.cosine_sim,
            "roi_rmse": self.roi_rmse,
            "edge_rmse": self.edge_rmse
        }


def calculate_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)
    
    物理意义：衡量整体像素级重建质量
    
    :param pred: 预测图像 [0, 1] 或 [0, 255]
    :param target: 目标图像
    :param max_val: 像素最大值
    :return: PSNR (dB)
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def calculate_ssim(
    pred: np.ndarray, 
    target: np.ndarray,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2
) -> float:
    """
    计算 SSIM (Structural Similarity Index)
    
    物理意义：衡量结构保持能力（边缘、纹理）
    参考：Wang et al., "Image Quality Assessment" (TIP 2004)
    
    :param pred: 预测图像 (H, W) 或 (H, W, C)
    :param target: 目标图像
    :param window_size: 滑窗大小
    :param C1, C2: 稳定常数
    :return: SSIM [0, 1]
    """
    # 转换为灰度（如果是彩色图像）
    if pred.ndim == 3:
        pred = np.mean(pred, axis=2)
        target = np.mean(target, axis=2)
    
    # 简化版 SSIM（全局计算，非滑窗）
    mu_x = np.mean(pred)
    mu_y = np.mean(target)
    sigma_x = np.std(pred)
    sigma_y = np.std(target)
    sigma_xy = np.mean((pred - mu_x) * (target - mu_y))
    
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
    
    return float(ssim)


def calculate_ssim_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    使用 PyTorch 计算 SSIM（支持批量）
    
    :param pred: (B, C, H, W) 或 (C, H, W)
    :param target: 同上
    :return: SSIM 均值
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = pred.mean(dim=[2, 3], keepdim=True)
    mu_y = target.mean(dim=[2, 3], keepdim=True)
    
    sigma_x_sq = ((pred - mu_x) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_y_sq = ((target - mu_y) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=[2, 3], keepdim=True)
    
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return float(ssim_map.mean().item())


def calculate_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算 RMSE (Root Mean Square Error)
    
    物理意义：像素值绝对误差，与浓度计算直接相关
    
    :param pred: 预测图像
    :param target: 目标图像
    :return: RMSE
    """
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def calculate_cosine_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算 RGB 向量的余弦相似度
    
    物理意义：光谱保真度，验证 R/G/B 比值是否正确
    这是本项目的特色指标，与损失函数中的 Cosine Loss 呼应
    
    :param pred: 预测图像 (H, W, 3)
    :param target: 目标图像 (H, W, 3)
    :return: 平均余弦相似度 [0, 1]
    """
    # 展平为 (N, 3)
    pred_flat = pred.reshape(-1, 3)
    target_flat = target.reshape(-1, 3)
    
    # 计算每个像素的余弦相似度
    dot_product = np.sum(pred_flat * target_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1) + 1e-8
    norm_target = np.linalg.norm(target_flat, axis=1) + 1e-8
    
    cos_sim = dot_product / (norm_pred * norm_target)
    
    return float(np.mean(cos_sim))


def calculate_roi_metrics(
    pred: np.ndarray, 
    target: np.ndarray, 
    roi_radius: int = 20
) -> Tuple[float, float]:
    """
    分区评估：ROI区域 vs 边缘区域
    
    物理意义：验证 ROI 加权损失的有效性
    预期：ROI-RMSE < Edge-RMSE（核心区精度更高）
    
    :param pred: 预测图像 (H, W, 3)
    :param target: 目标图像 (H, W, 3)
    :param roi_radius: ROI 区域半径（像素）
    :return: (roi_rmse, edge_rmse)
    """
    h, w = pred.shape[:2]
    center_y, center_x = h // 2, w // 2
    
    # 创建距离矩阵
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    
    # ROI 区域掩码
    roi_mask = dist <= roi_radius
    edge_mask = ~roi_mask
    
    # 分区计算 RMSE
    diff_sq = (pred - target) ** 2
    
    if roi_mask.sum() > 0:
        roi_rmse = float(np.sqrt(np.mean(diff_sq[roi_mask])))
    else:
        roi_rmse = 0.0
    
    if edge_mask.sum() > 0:
        edge_rmse = float(np.sqrt(np.mean(diff_sq[edge_mask])))
    else:
        edge_rmse = 0.0
    
    return roi_rmse, edge_rmse


def evaluate_batch(
    preds: np.ndarray,
    targets: np.ndarray,
    roi_radius: int = 20
) -> EvaluationMetrics:
    """
    批量评估
    
    :param preds: 预测图像 (N, H, W, 3) 或 (H, W, 3)
    :param targets: 目标图像
    :param roi_radius: ROI 区域半径
    :return: EvaluationMetrics
    """
    # 确保是批量格式
    if preds.ndim == 3:
        preds = preds[np.newaxis, ...]
        targets = targets[np.newaxis, ...]
    
    n_samples = preds.shape[0]
    
    # 累积指标
    psnr_sum = 0.0
    ssim_sum = 0.0
    rmse_sum = 0.0
    cos_sum = 0.0
    roi_rmse_sum = 0.0
    edge_rmse_sum = 0.0
    
    for i in range(n_samples):
        pred = preds[i]
        target = targets[i]
        
        psnr_sum += calculate_psnr(pred, target)
        ssim_sum += calculate_ssim(pred, target)
        rmse_sum += calculate_rmse(pred, target)
        cos_sum += calculate_cosine_similarity(pred, target)
        
        roi_rmse, edge_rmse = calculate_roi_metrics(pred, target, roi_radius)
        roi_rmse_sum += roi_rmse
        edge_rmse_sum += edge_rmse
    
    return EvaluationMetrics(
        psnr=psnr_sum / n_samples,
        ssim=ssim_sum / n_samples,
        rmse=rmse_sum / n_samples,
        cosine_sim=cos_sum / n_samples,
        roi_rmse=roi_rmse_sum / n_samples,
        edge_rmse=edge_rmse_sum / n_samples
    )


def evaluate_tensor_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    roi_radius: int = 20
) -> EvaluationMetrics:
    """
    评估 PyTorch Tensor 批量
    
    :param preds: (B, C, H, W) 格式的预测
    :param targets: (B, C, H, W) 格式的目标
    :param roi_radius: ROI 区域半径
    :return: EvaluationMetrics
    """
    # 转换为 numpy (B, H, W, C)
    preds_np = preds.permute(0, 2, 3, 1).detach().cpu().numpy()
    targets_np = targets.permute(0, 2, 3, 1).detach().cpu().numpy()
    
    return evaluate_batch(preds_np, targets_np, roi_radius)
