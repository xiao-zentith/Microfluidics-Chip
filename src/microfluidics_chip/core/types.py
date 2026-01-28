"""
核心数据类型定义
遵循 v1.1 强制规范：
- P0: 强类型接口锁死
- P1: Output对象路径字段使用str（相对路径）
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


# ==================== P0: 强类型定义 ====================

class ChamberDetection(BaseModel):
    """
    单个腔室检测结果（P0强制返回类型）
    ⚠️ detector.detect() 必须返回 List[ChamberDetection]
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]      # (cx, cy)
    class_id: int                    # 0=blank, 1=lit
    confidence: float                # 检测置信度 [0, 1]


class TransformParams(BaseModel):
    """
    几何变换参数（P0强制返回四元组的一部分）
    """
    rotation_angle: float              # 旋转角度（度）
    scale_factor: float                # 缩放系数
    chip_centroid: Tuple[float, float] # 芯片重心 (cx, cy)
    blank_arm_index: int               # 空白臂索引 [0, 3]
    
    # 可选：3x3变换矩阵（用于复现）
    matrix: Optional[List[List[float]]] = None


# ==================== Stage1 契约 ====================

class Stage1Result(BaseModel):
    """
    Stage1 内存结果（inference返回）
    ⚠️ 包含np.ndarray，不可JSON序列化
    用途：算法层返回，pipelines层消费
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    chip_id: str
    aligned_image: Any  # np.ndarray (H, W, 3) uint8
    chamber_slices: Any  # np.ndarray (12, H, W, 3) uint8
    transform_params: TransformParams
    chambers: List[ChamberDetection]  # P0: 强类型列表
    
    # 可选字段
    gt_slices: Optional[Any] = None  # np.ndarray (12, H, W, 3) uint8
    debug_vis: Optional[Any] = None  # np.ndarray
    processing_time: float = 0.0


class Stage1Output(BaseModel):
    """
    Stage1 落盘结果（pipelines保存，P1/P2规范）
    ✅ 只包含路径+元数据，可JSON序列化
    ⚠️ P1: 所有路径字段必须使用 str 类型（相对路径）
    ⚠️ P2: 路径值必须为固定文件名
    """
    chip_id: str
    
    # P1: 使用 str 类型（相对路径）
    # P2: 固定文件名
    aligned_image_path: str = "aligned.png"
    chamber_slices_path: str = "chamber_slices.npz"
    debug_vis_path: Optional[str] = "debug_visualization.png"
    
    transform_params: TransformParams
    num_chambers: int
    processing_time: float = 0.0
    has_gt_slices: bool = False


# ==================== Stage2 契约 ====================

class Stage2Result(BaseModel):
    """Stage2 内存结果"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    chip_id: str
    corrected_slices: Any  # np.ndarray (12, H, W, 3) uint8
    
    correction_params: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    processing_time: float = 0.0


class Stage2Output(BaseModel):
    """
    Stage2 落盘结果（P1/P2规范）
    ⚠️ P1: 路径字段使用 str
    """
    chip_id: str
    
    # P1: 使用 str 类型
    # P2: 固定文件名
    corrected_slices_path: str = "corrected_slices.npz"
    
    correction_params: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    processing_time: float = 0.0


# ==================== 转换函数 ====================

def stage1_result_to_output(
    result: Stage1Result,
    run_dir: "Path"  # 类型提示用引号避免循环导入
) -> Stage1Output:
    """
    将Stage1Result（内存）转为Stage1Output（落盘）
    ⚠️ 调用此函数前必须已保存npz/图像文件
    
    :param result: Stage1内存结果
    :param run_dir: 运行目录（用于日志，但不直接使用）
    :return: Stage1Output（可JSON序列化）
    """
    return Stage1Output(
        chip_id=result.chip_id,
        aligned_image_path="aligned.png",  # P2: 固定名称
        chamber_slices_path="chamber_slices.npz",
        debug_vis_path="debug_visualization.png" if result.debug_vis is not None else None,
        transform_params=result.transform_params,
        num_chambers=len(result.chambers),
        processing_time=result.processing_time,
        has_gt_slices=(result.gt_slices is not None)
    )


def stage2_result_to_output(
    result: Stage2Result,
    run_dir: "Path"
) -> Stage2Output:
    """将Stage2Result转为Stage2Output"""
    return Stage2Output(
        chip_id=result.chip_id,
        corrected_slices_path="corrected_slices.npz",  # P2: 固定名称
        correction_params=result.correction_params,
        metrics=result.metrics,
        processing_time=result.processing_time
    )


# ==================== 辅助工具 ====================

def validate_stage1_output_structure(run_dir: "Path") -> bool:
    """
    验证 Stage1 输出目录是否符合 P2 规范
    
    :param run_dir: Stage1运行目录
    :return: 是否符合规范
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    
    required_files = [
        "stage1_metadata.json",
        "chamber_slices.npz",
        "aligned.png"
    ]
    return all((run_dir / f).exists() for f in required_files)


def validate_stage2_output_structure(run_dir: "Path") -> bool:
    """
    验证 Stage2 输出目录是否符合 P2 规范
    
    :param run_dir: Stage2运行目录
    :return: 是否符合规范
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    
    required_files = [
        "stage2_metadata.json",
        "corrected_slices.npz"
    ]
    return all((run_dir / f).exists() for f in required_files)
