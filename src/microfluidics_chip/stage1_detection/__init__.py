"""
Stage1 Detection Module

包含：
- detector: YOLO 腔室检测器
- geometry_engine: 十字几何校正引擎
- inference: 推理入口
- synthesizer: 数据合成器
"""

from .detector import ChamberDetector
from .geometry_engine import CrossGeometryEngine
from .inference import infer_stage1
from .synthesizer import FullChipSynthesizer

__all__ = [
    "ChamberDetector",
    "CrossGeometryEngine",
    "infer_stage1",
    "FullChipSynthesizer",
]
