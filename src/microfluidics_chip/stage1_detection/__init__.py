"""
Stage1 Detection Module

包含：
- detector: YOLO 腔室检测器
- geometry_engine: 十字几何校正引擎
- inference: 推理入口 (含自适应推理)
- synthesizer: 数据合成器
- adaptive_detector: 自适应粗到精检测器
- topology_fitter: 拓扑约束模板拟合器
- preprocess: 统一预处理函数
"""

from .detector import ChamberDetector
from .geometry_engine import CrossGeometryEngine
from .inference import infer_stage1
from .synthesizer import FullChipSynthesizer

# 新增自适应检测模块
from .adaptive_detector import AdaptiveDetector, AdaptiveDetectionConfig, ClusterResult
from .topology_fitter import TopologyFitter, TopologyConfig, FittingResult
from .preprocess import apply_clahe, apply_invert, preprocess_image

# 从 inference 导入自适应入口 (将在后续添加)
# from .inference import infer_stage1_adaptive

__all__ = [
    # 原有导出
    "ChamberDetector",
    "CrossGeometryEngine",
    "infer_stage1",
    "FullChipSynthesizer",
    # 新增导出
    "AdaptiveDetector",
    "AdaptiveDetectionConfig",
    "ClusterResult",
    "TopologyFitter",
    "TopologyConfig",
    "FittingResult",
    "apply_clahe",
    "apply_invert",
    "preprocess_image",
]
