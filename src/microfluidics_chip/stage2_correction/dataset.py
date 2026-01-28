"""
微流控芯片数据集
继承自 v1.0 的 unet/model/train.py::MicrofluidicDataset

数据格式：
- npz 文件包含三个 key:
  - 'target_in': 待校正图像 (N, H, W, 3) float32 [0, 1]
  - 'ref_in': 参考图像 (N, H, W, 3) float32 [0, 1]
  - 'labels': 真值图像 (N, H, W, 3) float32 [0, 1]
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Literal


class MicrofluidicDataset(Dataset):
    """
    微流控芯片数据集
    
    从 npz 文件加载合成数据，自动划分训练/验证集
    
    :param npz_path: npz 数据文件路径
    :param mode: 'train' 或 'val'
    :param split_ratio: 训练集比例（默认0.9）
    """
    
    def __init__(
        self,
        npz_path: Path,
        mode: Literal['train', 'val'] = 'train',
        split_ratio: float = 0.9
    ):
        super().__init__()
        
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"找不到数据集文件: {npz_path}")
        
        print(f"[*] Loading data from {npz_path} ...")
        # allow_pickle=True 以防万一
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
        
        print(f"[{mode.upper()}] Dataset ready: {len(self.indices)} samples.")
    
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
        img_signal = self.target_in[real_idx]
        img_ref = self.ref_in[real_idx]
        img_gt = self.labels[real_idx]
        
        # 转换为 PyTorch Tensor 并调整维度 (H, W, 3) -> (3, H, W)
        # 注意：合成数据已经是 float32 (0-1)，不需要再除以 255
        signal_tensor = torch.from_numpy(img_signal).permute(2, 0, 1)
        ref_tensor = torch.from_numpy(img_ref).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(img_gt).permute(2, 0, 1)
        
        return signal_tensor, ref_tensor, gt_tensor
