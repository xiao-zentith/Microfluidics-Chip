"""
Stage2 Correction Module

包含：
- models: UNet 模型定义
- losses: 损失函数
- dataset: 数据集
- trainer: 训练器
- inference: 推理入口
"""

from .models.dual_stream_unet import RefGuidedUNet
from .losses import ROIWeightedLoss
from .dataset import MicrofluidicDataset
from .inference import infer_stage2

__all__ = [
    "RefGuidedUNet",
    "ROIWeightedLoss",
    "MicrofluidicDataset",
    "infer_stage2",
]
