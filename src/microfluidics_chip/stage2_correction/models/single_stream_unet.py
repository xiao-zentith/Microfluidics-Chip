"""
单流 UNet 模型 (消融实验A Baseline)

与双流 RefGuidedUNet 的区别：
- 移除 Reference Encoder 分支
- 移除 Feature Fusion 层
- 仅使用 Signal Encoder + Decoder

用途：验证 Reference Stream 的必要性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """基础卷积块: (Conv3x3 -> BN -> GELU) * 2"""
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
        # 尺寸校验
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SingleStreamUNet(nn.Module):
    """
    单流 UNet (消融实验A Baseline)
    
    与 RefGuidedUNet 的区别：
    - 无 Reference Encoder
    - 无 Feature Fusion
    - 仅 Signal Encoder + Decoder
    
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
        
        # --- Encoder ---
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # --- Bottleneck (无 Fusion，直接传递) ---
        # 保持与双流模型相同的瓶颈维度
        self.bottleneck = DoubleConv(features[3], features[3])
        
        # --- Decoder ---
        self.up1 = Up(features[3] + features[2], features[2])
        self.up2 = Up(features[2] + features[1], features[1])
        self.up3 = Up(features[1] + features[0], features[0])
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x_signal: torch.Tensor, x_ref: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        :param x_signal: 信号流输入 (B, 3, H, W)
        :param x_ref: 参考流输入 (B, 3, H, W) - **忽略，仅保持接口兼容**
        :return: 校正后的输出 (B, 3, H, W)
        
        注意：x_ref 参数被忽略，仅为保持与 RefGuidedUNet 相同的调用接口
        """
        # 1. 编码
        x1 = self.inc(x_signal)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 2. 瓶颈（无 Reference Fusion）
        bottleneck = self.bottleneck(x4)
        
        # 3. 解码
        x = self.up1(bottleneck, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        return torch.sigmoid(self.outc(x))
    
    def get_num_params(self) -> int:
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters())
