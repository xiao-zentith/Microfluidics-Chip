"""
Microfluidics Chip Processing Pipeline
微流控芯片图像处理流水线

包含三个主要阶段：
- Stage1: 目标检测与几何校正（YOLO + GeometryEngine）
- Stage2: 光照校正（双流UNet）
- Stage3: 浓度提取（预留）
"""

__version__ = "0.1.0"
__author__ = "Microfluidics Team"

from .core.types import (
    ChamberDetection,
    TransformParams,
    Stage1Result,
    Stage1Output,
    Stage2Result,
    Stage2Output,
)

__all__ = [
    "ChamberDetection",
    "TransformParams",
    "Stage1Result",
    "Stage1Output",
    "Stage2Result",
    "Stage2Output",
    "__version__",
]
