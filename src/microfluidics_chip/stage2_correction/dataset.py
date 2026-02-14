"""
微流控芯片数据集 (v1.2)

数据格式：
- npz 文件包含三个 key:
  - 'target_in': 待校正图像 (N, H, W, 3) float32 [0, 1]
  - 'ref_in': 参考图像 (N, H, W, 3) float32 [0, 1]
  - 'labels': 真值图像 (N, H, W, 3) float32 [0, 1]

v1.2 更新:
- 在线增强简化为仅几何变换（光学增强已移至离线阶段）
- 移除 aug_intensity 参数
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Literal, Optional

from .augmentations import GeometricAugmentation, get_train_augmentation, get_val_augmentation


class MicrofluidicDataset(Dataset):
    """
    微流控芯片数据集
    
    从 npz 文件加载数据，自动划分训练/验证集
    
    v1.2 方案:
    - 在线增强仅包含几何变换（翻转、90°旋转）
    - 光学增强在离线数据准备阶段完成
    
    :param npz_path: npz 数据文件路径
    :param mode: 'train' 或 'val'
    :param split_ratio: 训练集比例（默认0.9）
    :param augment: 是否启用在线几何增强（仅对训练集有效）
    :param rotate90: 是否启用90°旋转（若使用位置编码建议关闭）
    """
    
    def __init__(
        self,
        npz_path: Path,
        mode: Literal['train', 'val'] = 'train',
        split_ratio: float = 0.9,
        augment: bool = False,
        rotate90: bool = True,
        # 向后兼容，已废弃
        aug_intensity: float = None
    ):
        super().__init__()
        
        # 向后兼容警告
        if aug_intensity is not None:
            import warnings
            warnings.warn(
                "aug_intensity parameter is deprecated in v1.2. "
                "Optical augmentation has been moved to offline stage. "
                "This parameter will be ignored.",
                DeprecationWarning
            )
        
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"找不到数据集文件: {npz_path}")
        
        print(f"[*] Loading data from {npz_path} ...")
        data = np.load(npz_path, allow_pickle=True)
        
        # 读取数据 (N, H, W, 3) -> float32 [0, 1]
        self.target_in = data['target_in']  # 待校正图像
        self.ref_in = data['ref_in']        # 参考图像
        self.labels = data['labels']        # 真值图像
        
        total = len(self.target_in)
        split_idx = int(total * split_ratio)
        
        # 划分训练/验证集
        if mode == 'train':
            self.indices = range(0, split_idx)
        else:
            self.indices = range(split_idx, total)
        
        # 设置数据增强 (v1.2: 仅几何增强)
        self.mode = mode
        self.augment = augment and mode == 'train'  # 仅训练集增强
        self.transform = None
        
        if self.augment:
            self.transform = get_train_augmentation(rotate90=rotate90)
            print(f"[{mode.upper()}] Online geometric augmentation ENABLED (v1.2)")
            print(f"  - Flip: horizontal/vertical (p=0.5)")
            print(f"  - Rotate90: {'enabled' if rotate90 else 'disabled'}")
        
        aug_status = "geometric-augmented" if self.augment else "original"
        print(f"[{mode.upper()}] Dataset ready: {len(self.indices)} samples ({aug_status})")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        :param idx: 索引
        :return: (signal_tensor, ref_tensor, gt_tensor)
                 每个都是 (3, H, W) float32 tensor
        """
        real_idx = self.indices[idx]
        
        # 获取 numpy 数据 (H, W, 3)
        img_signal = self.target_in[real_idx].copy()
        img_ref = self.ref_in[real_idx].copy()
        img_gt = self.labels[real_idx].copy()
        
        # 在线几何增强（如果启用）
        if self.augment and self.transform is not None:
            img_signal, img_ref, img_gt = self.transform(img_signal, img_ref, img_gt)
        
        # 转换为 PyTorch Tensor 并调整维度 (H, W, 3) -> (3, H, W)
        signal_tensor = torch.from_numpy(img_signal).permute(2, 0, 1).float()
        ref_tensor = torch.from_numpy(img_ref).permute(2, 0, 1).float()
        gt_tensor = torch.from_numpy(img_gt).permute(2, 0, 1).float()
        
        return signal_tensor, ref_tensor, gt_tensor
