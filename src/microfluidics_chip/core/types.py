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
    quality_metrics: Optional[Dict[str, Any]] = None
    quality_gate_passed: Optional[bool] = None
    detection_mode: str = "standard"
    retry_attempt: int = 0


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
    geo_success: Optional[bool] = None
    geo_quality_level: Optional[str] = None
    semantic_ready: Optional[bool] = None
    used_fallback: Optional[bool] = None
    fallback_reason: Optional[str] = None
    blank_status: Optional[str] = None
    reference_arm_pred: Optional[str] = None
    blank_id_pred: Optional[int] = None
    reprojection_error_mean_px: Optional[float] = None
    reprojection_error_max_px: Optional[float] = None
    slice_center_offset_max_px: Optional[float] = None
    slice_mode: Optional[str] = None
    geometry_suspect: Optional[bool] = None
    model_chosen: Optional[str] = None
    fill_ratio: Optional[float] = None
    used_real_points: Optional[int] = None
    pitch_final: Optional[float] = None
    n_det_raw: Optional[int] = None
    n_det_dedup: Optional[int] = None
    blank_valid: Optional[bool] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    quality_gate_passed: Optional[bool] = None
    detection_mode: str = "standard"
    retry_attempt: int = 0


# ==================== 自适应检测结果 ====================

class AdaptiveDetectionResult(BaseModel):
    """
    自适应检测完整结果
    
    包含粗到精检测结果、聚类信息、拓扑拟合结果和暗腔室判定
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 检测结果
    detections: List[ChamberDetection]  # 最终检测列表
    
    # 聚类信息
    roi_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    cluster_score: float  # 聚类质量评分
    is_fallback: bool = False  # 是否使用回退策略
    
    # 拓扑拟合结果
    fitted_centers: Any  # np.ndarray (12, 2)
    visibility: Any  # np.ndarray (12,) bool
    detected_mask: Any  # np.ndarray (12,) bool
    dark_chamber_indices: List[int]  # 暗腔室索引列表
    
    # 质量指标
    inlier_ratio: float = 0.0  # RANSAC 内点比例
    reprojection_error: float = 0.0  # 平均重投影误差
    fit_success: bool = False  # 拓扑拟合是否成功
    
    # 元数据
    processing_time: float = 0.0


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
    qm = result.quality_metrics if isinstance(result.quality_metrics, dict) else {}
    geo_success = qm.get("geo_success")
    semantic_ready = qm.get("semantic_ready", qm.get("blank_pass"))
    used_fallback = qm.get("used_fallback")
    fallback_reason = qm.get("fallback_reason")
    geo_quality_level = qm.get("geo_quality_level")
    blank_status = qm.get("blank_status")
    reference_arm_pred = qm.get("reference_arm_pred", qm.get("reference_arm"))
    blank_id_pred = qm.get("blank_id_pred", qm.get("blank_idx"))
    reproj_mean = qm.get("reprojection_error_mean_px", qm.get("geometry_reprojection_error_mean_px"))
    reproj_max = qm.get("reprojection_error_max_px", qm.get("geometry_reprojection_error_max_px"))
    center_offset = qm.get("slice_center_offset_max_px", qm.get("geometry_slice_center_offset_max_px"))
    slice_mode = qm.get("slice_mode")
    geometry_suspect = qm.get("geometry_suspect")
    model_chosen = qm.get("model_chosen")
    fill_ratio = qm.get("fill_ratio")
    used_real_points = qm.get("used_real_points")
    pitch_final = qm.get("pitch_final", qm.get("pitch_px"))
    n_det_raw = qm.get("n_det_raw")
    n_det_dedup = qm.get("n_det_dedup")
    blank_valid = qm.get("blank_valid")
    return Stage1Output(
        chip_id=result.chip_id,
        aligned_image_path="aligned.png",  # P2: 固定名称
        chamber_slices_path="chamber_slices.npz",
        debug_vis_path="debug_visualization.png" if result.debug_vis is not None else None,
        transform_params=result.transform_params,
        num_chambers=len(result.chambers),
        processing_time=result.processing_time,
        has_gt_slices=(result.gt_slices is not None),
        geo_success=(None if geo_success is None else bool(geo_success)),
        geo_quality_level=(None if geo_quality_level is None else str(geo_quality_level)),
        semantic_ready=(None if semantic_ready is None else bool(semantic_ready)),
        used_fallback=(None if used_fallback is None else bool(used_fallback)),
        fallback_reason=(None if fallback_reason is None else str(fallback_reason)),
        blank_status=(None if blank_status is None else str(blank_status)),
        reference_arm_pred=(None if reference_arm_pred is None else str(reference_arm_pred)),
        blank_id_pred=(None if blank_id_pred is None else int(blank_id_pred)),
        reprojection_error_mean_px=(None if reproj_mean is None else float(reproj_mean)),
        reprojection_error_max_px=(None if reproj_max is None else float(reproj_max)),
        slice_center_offset_max_px=(None if center_offset is None else float(center_offset)),
        slice_mode=(None if slice_mode is None else str(slice_mode)),
        geometry_suspect=(None if geometry_suspect is None else bool(geometry_suspect)),
        model_chosen=(None if model_chosen is None else str(model_chosen)),
        fill_ratio=(None if fill_ratio is None else float(fill_ratio)),
        used_real_points=(None if used_real_points is None else int(used_real_points)),
        pitch_final=(None if pitch_final is None else float(pitch_final)),
        n_det_raw=(None if n_det_raw is None else int(n_det_raw)),
        n_det_dedup=(None if n_det_dedup is None else int(n_det_dedup)),
        blank_valid=(None if blank_valid is None else bool(blank_valid)),
        quality_metrics=result.quality_metrics,
        quality_gate_passed=result.quality_gate_passed,
        detection_mode=result.detection_mode,
        retry_attempt=result.retry_attempt
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
