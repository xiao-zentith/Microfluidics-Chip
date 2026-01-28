"""
双流 UNet 模型
继承自 v1.0 的 unet/model/unet.py::RefGuidedUNet
100% 保留原始架构和参数

论文核心模型：Dual-Stream U-Net
- Stream 1: Signal Encoder（信号流编码器）
- Stream 2: Reference Encoder（参考流编码器）
- Fusion: Feature-level Fusion（特征融合）
- Decoder: Skip-connected Decoder（跳跃连接解码器）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """
    基础卷积块: (Conv3x3 -> BN -> GELU) * 2
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # 尺寸校验（防止奇数尺寸导致的拼接错误）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # x1: Decoder feature, x2: Encoder Skip Connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class EnvironmentEncoder(nn.Module):
    """
    分支2：环境参考编码器（Reference Stream）
    将参考图像编码为全局特征向量
    """
    def __init__(self, in_channels: int = 3, features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.inc = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList()
        in_ch = features[0]
        for feat in features[1:]:
            self.downs.append(Down(in_ch, feat))
            in_ch = feat
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        for down in self.downs:
            x = down(x)
        return self.global_pool(x)


class RefGuidedUNet(nn.Module):
    """
    论文核心模型: Dual-Stream U-Net
    
    架构：
    1. Signal Encoder: 提取待校正图像的多尺度特征
    2. Reference Encoder: 提取参考图像的全局环境特征
    3. Feature Fusion: 在瓶颈层融合两路特征
    4. Decoder: 通过跳跃连接恢复空间细节
    
    :param in_channels: 输入通道数（默认3，RGB）
    :param out_channels: 输出通道数（默认3，RGB）
    :param features: 特征通道数列表
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: List[int] = [64, 128, 256, 512]
    ):
        super().__init__()
        
        # --- Stream 1: Signal Encoder ---
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # --- Stream 2: Reference Encoder ---
        self.ref_encoder = EnvironmentEncoder(in_channels, features)
        
        # --- Fusion ---
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(features[3] * 2, features[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(features[3]),
            nn.GELU()
        )
        
        # --- Decoder ---
        self.up1 = Up(features[3] + features[2], features[2])
        self.up2 = Up(features[2] + features[1], features[1])
        self.up3 = Up(features[1] + features[0], features[0])
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(
        self,
        x_signal: torch.Tensor,
        x_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        :param x_signal: 信号流输入 (B, 3, H, W)
        :param x_ref: 参考流输入 (B, 3, H, W)
        :return: 校正后的输出 (B, 3, H, W)
        """
        # 1. 信号流编码
        x1 = self.inc(x_signal)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 2. 参考流编码
        env_vec = self.ref_encoder(x_ref)
        
        # 3. 特征融合
        env_map = env_vec.expand_as(x4)
        fused = torch.cat([x4, env_map], dim=1)
        bottleneck = self.fusion_conv(fused)
        
        # 4. 解码
        x = self.up1(bottleneck, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        return torch.sigmoid(self.outc(x))
