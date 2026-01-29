"""
Stage2 UNet 在线数据增强模块 (v1.2)

根据 Final v1.2 方案:
- 在线增强仅包含几何变换（翻转、90°旋转）
- 光学增强和噪声增强移至离线阶段（prepare_training_data.py）
- 确保三元组 (signal, ref, gt) 同步变换

设计原则:
1. 物理一致性: 光照增强在全图切片前应用（离线）
2. 防泄漏: 在线增强不破坏signal与ref的光照一致性
3. 简洁性: 仅同步几何变换
"""

import numpy as np
from typing import Tuple, Optional
import random


class GeometricAugmentation:
    """
    仅几何增强（在线使用）
    
    v1.2 方案: 光学增强已移至离线阶段
    在线阶段仅允许三元组同步的几何变换
    """
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate90: bool = True
    ):
        """
        :param flip_prob: 翻转概率 (水平和垂直各自独立)
        :param rotate90: 是否启用90°旋转 (若使用位置编码建议关闭)
        """
        self.flip_prob = flip_prob
        self.rotate90 = rotate90
    
    def __call__(
        self,
        signal: np.ndarray,
        ref: np.ndarray,
        gt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        应用同步几何增强
        
        :param signal: 待校正图像 (H, W, 3) float32 [0, 1]
        :param ref: 参考图像 (H, W, 3) float32 [0, 1]
        :param gt: Ground Truth (H, W, 3) float32 [0, 1]
        :return: 增强后的 (signal, ref, gt)
        """
        # 水平翻转
        if random.random() < self.flip_prob:
            signal = np.fliplr(signal).copy()
            ref = np.fliplr(ref).copy()
            gt = np.fliplr(gt).copy()
        
        # 垂直翻转
        if random.random() < self.flip_prob:
            signal = np.flipud(signal).copy()
            ref = np.flipud(ref).copy()
            gt = np.flipud(gt).copy()
        
        # 90°旋转 (k次90度)
        if self.rotate90:
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                signal = np.rot90(signal, k).copy()
                ref = np.rot90(ref, k).copy()
                gt = np.rot90(gt, k).copy()
        
        return signal, ref, gt


# ========== 便捷函数 ==========

def get_train_augmentation(rotate90: bool = True) -> GeometricAugmentation:
    """
    获取训练用的在线增强器
    
    :param rotate90: 是否启用90°旋转 (若使用位置编码建议关闭)
    :return: 增强器
    """
    return GeometricAugmentation(
        flip_prob=0.5,
        rotate90=rotate90
    )


def get_val_augmentation() -> Optional[GeometricAugmentation]:
    """
    验证集不使用增强
    """
    return None


# ========== 离线增强函数 (供 prepare_training_data.py 调用) ==========

def apply_shot_noise(image: np.ndarray, peak_photon_count: int) -> np.ndarray:
    """
    光子计数Shot Noise模型 (v1.2)
    
    物理模型: I_noisy = Poisson(I_clean * N_peak) / N_peak
    
    :param image: 输入图像 float32 [0, 1]
    :param peak_photon_count: 峰值光子计数 [30, 100]
                              30 = 低光照 (噪声明显)
                              100 = 正常光照 (噪声轻微)
    :return: 添加噪声后的图像
    """
    # 模拟光子采样
    photon_image = image * peak_photon_count
    # Poisson采样
    noisy_photons = np.random.poisson(photon_image.astype(np.float64))
    # 归一化回 [0, 1]
    noisy_image = noisy_photons / peak_photon_count
    return np.clip(noisy_image, 0, 1).astype(np.float32)


def apply_isp_degradation(
    image: np.ndarray,
    illum_strength: float = 0.3,
    wb_range: Tuple[float, float] = (0.9, 1.1),
    exposure_range: Tuple[float, float] = (0.85, 1.15),
    gamma_range: Tuple[float, float] = (0.8, 1.2),
    peak_photon_range: Tuple[int, int] = (30, 100)
) -> np.ndarray:
    """
    非线性ISP退化链 (v1.2)
    
    I_aug = Gamma(Exposure(WhiteBal(I_raw * M_illum))) + N_shot
    
    用于离线全图增强，在切片前应用
    
    :param image: 输入全图 uint8 [0, 255] 或 float32 [0, 1]
    :param illum_strength: 光照场强度
    :param wb_range: 白平衡增益范围
    :param exposure_range: 曝光增益范围
    :param gamma_range: Gamma值范围
    :param peak_photon_range: 峰值光子计数范围
    :return: 退化后的图像 (与输入相同格式)
    """
    # 统一转换为 float32 [0, 1]
    is_uint8 = image.dtype == np.uint8
    if is_uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.copy()
    
    h, w = img.shape[:2]
    
    # 1. 空间光照场 M_illum
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    angle = np.deg2rad(random.uniform(0, 360))
    
    # 方向性梯度
    gradient = X * np.cos(angle) + Y * np.sin(angle)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    
    # 径向渐晕
    cx, cy = w // 2, h // 2
    radial = np.sqrt((X - cx)**2 + (Y - cy)**2)
    radial = radial / (radial.max() + 1e-6)
    
    # 混合
    mix = random.random()
    field = mix * gradient + (1 - mix) * radial
    illum_map = 1.0 - illum_strength * field
    img = img * illum_map[..., np.newaxis]
    
    # 2. 白平衡漂移 (R/B通道)
    r_gain = random.uniform(*wb_range)
    b_gain = random.uniform(*wb_range)
    img[:, :, 2] *= r_gain  # R通道 (BGR格式)
    img[:, :, 0] *= b_gain  # B通道
    
    # 3. 曝光增益
    exposure_gain = random.uniform(*exposure_range)
    img = img * exposure_gain
    
    # 4. Gamma校正
    gamma = random.uniform(*gamma_range)
    img = np.clip(img, 0, 1)
    img = np.power(img, gamma)
    
    # 5. Shot Noise (光子计数模型)
    peak_photon = random.randint(*peak_photon_range)
    img = apply_shot_noise(img, peak_photon)
    
    # 转换回原始格式
    if is_uint8:
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        return np.clip(img, 0, 1).astype(np.float32)


# ========== 向后兼容 ==========

# 旧API兼容 (已废弃，保留以避免破坏现有代码)
class MicrofluidicAugmentation(GeometricAugmentation):
    """
    向后兼容的别名
    
    警告: 此类已重命名为 GeometricAugmentation
    光学增强已移至离线阶段
    """
    def __init__(
        self,
        geometric_prob: float = 0.5,
        optical_prob: float = 0.8,  # 已忽略
        noise_prob: float = 0.5,    # 已忽略
        intensity: float = 0.3      # 已忽略
    ):
        import warnings
        warnings.warn(
            "MicrofluidicAugmentation is deprecated. "
            "Use GeometricAugmentation for online augmentation. "
            "Optical augmentation has been moved to offline stage.",
            DeprecationWarning
        )
        super().__init__(flip_prob=geometric_prob, rotate90=True)
