"""
ROI 加权损失函数
继承自 v1.0 的 unet/model/unet.py::ROIWeightedLoss

核心思想：
1. 中心反应区（ROI）权重高（1.0）
2. 边缘/背景区域权重低（0.1），容忍对齐误差
3. 混合损失：像素损失 + 余弦相似度损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIWeightedLoss(nn.Module):
    """
    ROI 加权损失函数
    
    损失组成：
    1. ROI-Weighted MSE Loss: 光度/强度准确性
       - ROI 区域权重 = 1.0
       - 边缘/背景权重 = edge_weight（默认0.1）
    
    2. Cosine Similarity Loss: 光谱/浓度准确性
       - 保证 RGB 向量方向一致（R/G 比值正确）
    
    :param roi_radius: 核心反应区半径（像素）
    :param edge_weight: 边缘/背景区域权重
    :param lambda_cos: 余弦相似度损失权重
    """
    
    def __init__(
        self,
        roi_radius: int = 20,
        edge_weight: float = 0.1,
        lambda_cos: float = 0.2
    ):
        """
        初始化损失函数
        
        Args:
            roi_radius: 核心反应区半径（像素），该区域权重为 1.0
                       注意：如果切片变大了，只要腔室没变大，这个值通常不用变
            edge_weight: 边缘/背景区域的权重（0.1），容忍对齐误差
            lambda_cos: 余弦相似度损失的权重
        """
        super().__init__()
        self.roi_radius = roi_radius
        self.edge_weight = edge_weight
        self.lambda_cos = lambda_cos
        
        # 采用缓存机制（动态生成权重图）
        self.weight_map = None
        self.current_size = None
    
    def _create_weight_map(
        self,
        size: int,
        r: int,
        w_edge: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        动态生成权重图
        
        :param size: 图像尺寸（假设方形 H=W）
        :param r: ROI 半径
        :param w_edge: 边缘权重
        :param device: 设备
        :return: 权重图 (1, 1, H, W)
        """
        center = size // 2
        Y, X = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing='ij'
        )
        dist = torch.sqrt((X - center)**2 + (Y - center)**2)
        
        # ROI区域权重=1.0，边缘权重=w_edge
        mask = torch.where(
            dist <= r,
            torch.tensor(1.0, device=device),
            torch.tensor(w_edge, device=device)
        )
        
        # 扩展维度 [H, W] -> [1, 1, H, W] 以进行广播
        return mask.view(1, 1, size, size)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        :param pred: 预测图像 (B, 3, H, W)
        :param target: 目标图像 (B, 3, H, W)
        :return: (total_loss, loss_pixel, loss_cos)
        """
        # 获取当前输入的尺寸（假设是方形 H=W）
        batch, channel, h, w = pred.shape
        
        # 如果尺寸变了，或者还没生成过 map，就重新生成
        if (self.weight_map is None or
            self.current_size != h or
            self.weight_map.device != pred.device):
            
            self.current_size = h
            self.weight_map = self._create_weight_map(
                h, self.roi_radius, self.edge_weight, pred.device
            )
        
        # 1. ROI Weighted MSE Loss（光度/强度准确性）
        # 使用 MSE 而不是 L1，因为 MSE 对大误差（光照梯度）惩罚更重
        loss_pixel = torch.mean(self.weight_map * (pred - target) ** 2)
        
        # 2. Cosine Similarity Loss（光谱/浓度准确性）
        # 保证 RGB 向量的方向一致，即 R/G 比值正确
        cos_sim = F.cosine_similarity(pred, target, dim=1, eps=1e-8)
        loss_cos = 1.0 - cos_sim.mean()
        
        # 总损失
        total_loss = loss_pixel + self.lambda_cos * loss_cos
        
        return total_loss, loss_pixel, loss_cos
