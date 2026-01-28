"""
Stage2 推理入口（纯算法逻辑，无IO）

职责：
- 加载 UNet 模型
- 对切片进行光照校正
- 返回内存结果
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from ..core.types import Stage2Result
from ..core.config import Stage2Config
from ..core.logger import get_logger
from .models.dual_stream_unet import RefGuidedUNet

logger = get_logger("stage2_correction.inference")


def infer_stage2(
    chip_id: str,
    chamber_slices: np.ndarray,
    config: Stage2Config,
    model: Optional[RefGuidedUNet] = None
) -> Stage2Result:
    """
    Stage2 推理入口
    
    流程：
    1. 加载模型（如果未提供）
    2. 提取参考切片（第0个切片，Blank Arm）
    3. 对每个待校正切片进行推理
    4. 返回内存结果
    
    :param chip_id: 芯片ID
    :param chamber_slices: 切片数组 (12, H, W, 3) uint8
    :param config: Stage2 配置
    :param model: RefGuidedUNet 实例（批处理时复用）
    :return: Stage2Result（内存对象）
    """
    start_time = time.time()
    
    # ==================== 加载模型 ====================
    device = torch.device(config.model.device)
    
    if model is None:
        logger.info(f"[{chip_id}] Initializing UNet model...")
        model = RefGuidedUNet(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            features=config.model.features
        ).to(device)
        
        # 加载权重
        if config.weights_path:
            weights_path = Path(config.weights_path)
            if weights_path.exists():
                model.load_state_dict(torch.load(weights_path, map_location=device))
                logger.info(f"[{chip_id}] Loaded weights from {weights_path}")
            else:
                logger.warning(f"[{chip_id}] Weights file not found: {weights_path}")
        
        model.eval()
    
    # ==================== 数据准备 ====================
    # 检查输入
    if chamber_slices.shape[0] < 12:
        raise ValueError(f"Chip {chip_id}: Expected 12 slices, got {chamber_slices.shape[0]}")
    
    # 提取参考切片（第0个，Blank Arm）
    ref_slice = chamber_slices[0]  # (H, W, 3) uint8
    
    # 转换为 tensor (3, H, W) float32 [0, 1]
    ref_tensor = torch.from_numpy(ref_slice).permute(2, 0, 1).float() / 255.0
    ref_tensor = ref_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # ==================== 推理 ====================
    corrected_slices = []
    
    logger.info(f"[{chip_id}] Correcting {len(chamber_slices)} slices...")
    
    with torch.no_grad():
        for i, slice_img in enumerate(chamber_slices):
            # 转换为 tensor
            signal_tensor = torch.from_numpy(slice_img).permute(2, 0, 1).float() / 255.0
            signal_tensor = signal_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
            
            # 推理
            output = model(signal_tensor, ref_tensor)  # (1, 3, H, W)
            
            # 转换回 uint8 (H, W, 3)
            output_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
            output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
            
            corrected_slices.append(output_np)
    
    corrected_slices = np.array(corrected_slices)  # (12, H, W, 3)
    
    # ==================== 计算耗时 ====================
    processing_time = time.time() - start_time
    
    # ==================== 返回内存结果 ====================
    result = Stage2Result(
        chip_id=chip_id,
        corrected_slices=corrected_slices,
        processing_time=processing_time
    )
    
    logger.info(f"[{chip_id}] Stage2 inference complete in {processing_time:.2f}s")
    return result
