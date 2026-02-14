"""
Stage1 推理入口（纯算法逻辑，无IO）
遵循 v1.1 P3 强制规范：
- GT 处理时必须实例化独立的 geometry_engine

v2.0 新增：
- infer_stage1_adaptive: 自适应粗到精 + 拓扑兜底推理
"""

import time
import numpy as np
from typing import Optional, List, Dict, Any
from ..core.types import Stage1Result, ChamberDetection, AdaptiveDetectionResult
from ..core.config import Stage1Config, AdaptiveDetectionConfig, TopologyConfig
from ..core.logger import get_logger
from .detector import ChamberDetector
from .geometry_engine import CrossGeometryEngine
from .adaptive_detector import AdaptiveDetector, AdaptiveDetectionConfig as AdaptiveConfig
from .topology_fitter import TopologyFitter, TopologyConfig as TopoConfig

logger = get_logger("stage1_detection.inference")


def infer_stage1_from_detections(
    chip_id: str,
    raw_image: np.ndarray,
    gt_image: Optional[np.ndarray],
    detections_raw: List[ChamberDetection],
    config: Stage1Config,
    detector: Optional[ChamberDetector] = None,
    geometry_engine: Optional[CrossGeometryEngine] = None,
    quality_metrics: Optional[Dict[str, Any]] = None,
    quality_gate_passed: Optional[bool] = None,
    detection_mode: str = "standard",
    retry_attempt: int = 0
) -> Stage1Result:
    """
    使用已给定的检测结果执行 Stage1 几何校正与切片提取。

    该函数用于统一标准流程和自适应流程的后处理逻辑，确保：
    - 几何处理一致
    - GT 处理保持 P3（独立 geometry engine）
    - Result 输出字段一致（包含质量指标）
    """
    start_time = time.time()

    if len(detections_raw) < 12:
        raise ValueError(f"Chip {chip_id}: Only {len(detections_raw)} chambers detected")

    if geometry_engine is None:
        geometry_engine = CrossGeometryEngine(config.geometry)
        logger.info("Initialized new CrossGeometryEngine")

    # ==================== 几何校正 Raw 图像 ====================
    logger.info(f"[{chip_id}] Processing geometry for raw image...")
    aligned_image, chamber_slices, transform_params, debug_vis = geometry_engine.process(
        raw_image, detections_raw
    )

    if aligned_image is None:
        logger.error(f"[{chip_id}] Geometry processing failed")
        raise RuntimeError(f"Chip {chip_id}: Geometry processing failed")

    logger.info(f"[{chip_id}] Geometry processing complete: {len(chamber_slices)} slices extracted")

    # ==================== P3: GT 处理（独立引擎） ====================
    gt_slices = None
    if gt_image is not None:
        if detector is None:
            detector = ChamberDetector(config.yolo)

        logger.info(f"[{chip_id}] Processing GT image with INDEPENDENT engine (P3)")

        # P3 关键：实例化独立引擎
        gt_engine = CrossGeometryEngine(config.geometry)

        # 检测 GT
        detections_gt = detector.detect(gt_image)

        if len(detections_gt) >= 12:
            # 使用独立引擎处理
            _, gt_slices_array, _, _ = gt_engine.process(gt_image, detections_gt)

            if gt_slices_array is not None:
                gt_slices = gt_slices_array
                logger.info(f"[{chip_id}] GT processing complete: {len(gt_slices)} slices")
            else:
                logger.warning(f"[{chip_id}] GT geometry processing failed")
        else:
            logger.warning(f"[{chip_id}] Insufficient GT detections: {len(detections_gt)}")

    # ==================== 计算耗时 ====================
    processing_time = time.time() - start_time

    # ==================== 返回内存结果 ====================
    result = Stage1Result(
        chip_id=chip_id,
        aligned_image=aligned_image,
        chamber_slices=chamber_slices,
        transform_params=transform_params,
        chambers=detections_raw,
        gt_slices=gt_slices,
        debug_vis=debug_vis,
        processing_time=processing_time,
        quality_metrics=quality_metrics,
        quality_gate_passed=quality_gate_passed,
        detection_mode=detection_mode,
        retry_attempt=retry_attempt
    )

    logger.info(
        f"[{chip_id}] Stage1 inference ({detection_mode}) complete in {processing_time:.2f}s"
    )
    return result


def infer_stage1(
    chip_id: str,
    raw_image: np.ndarray,
    gt_image: Optional[np.ndarray],
    config: Stage1Config,
    detector: Optional[ChamberDetector] = None,
    geometry_engine: Optional[CrossGeometryEngine] = None
) -> Stage1Result:
    """
    Stage1 推理入口（依赖注入版）
    
    职责：
    1. 调用 detector 检测腔室
    2. 调用 geometry_engine 进行几何校正和切片
    3. 处理 GT 图像（P3：独立引擎）
    4. 返回内存结果（Stage1Result）
    
    P3 强制规范：
    - GT 处理时必须实例化独立的 CrossGeometryEngine
    - 防止 GT 污染 Raw 的 last_* 状态
    
    :param chip_id: 芯片ID
    :param raw_image: 原始图像 (H, W, 3) uint8
    :param gt_image: GT图像（可选）
    :param config: Stage1 配置
    :param detector: ChamberDetector 实例（批处理时复用）
    :param geometry_engine: CrossGeometryEngine 实例（批处理时复用）
    :return: Stage1Result（内存对象）
    """
    # ==================== 实例化（如果未提供） ====================
    if detector is None:
        detector = ChamberDetector(config.yolo)
        logger.info("Initialized new ChamberDetector")

    # ==================== 检测 Raw 图像 ====================
    logger.info(f"[{chip_id}] Detecting chambers in raw image...")
    detections_raw = detector.detect(raw_image)

    if len(detections_raw) < 12:
        logger.error(f"[{chip_id}] Insufficient detections: {len(detections_raw)} < 12")
        raise ValueError(f"Chip {chip_id}: Only {len(detections_raw)} chambers detected")

    logger.info(f"[{chip_id}] Detected {len(detections_raw)} chambers")

    return infer_stage1_from_detections(
        chip_id=chip_id,
        raw_image=raw_image,
        gt_image=gt_image,
        detections_raw=detections_raw,
        config=config,
        detector=detector,
        geometry_engine=geometry_engine,
        detection_mode="standard",
        retry_attempt=0
    )


# ==================== 自适应推理入口 (v2.0) ====================

def infer_stage1_adaptive(
    chip_id: str,
    raw_image: np.ndarray,
    config: Stage1Config,
    adaptive_config: Optional[AdaptiveDetectionConfig] = None,
    topology_config: Optional[TopologyConfig] = None,
    detector: Optional[ChamberDetector] = None
) -> AdaptiveDetectionResult:
    """
    自适应 Stage1 推理（粗到精 + 拓扑兜底）
    
    适用场景：
    - 暗腔室漏检
    - 远近尺度变化
    - 复杂光照环境
    
    流程：
    1. 自适应检测：global_scan -> cluster_roi -> fine_scan
    2. 拓扑拟合：RANSAC 模板匹配 + 缺失腔室回填
    3. 暗腔室判定：基于拓扑位置的亮度分析
    
    :param chip_id: 芯片 ID
    :param raw_image: 原始图像 (H, W, 3) BGR uint8
    :param config: Stage1 基础配置
    :param adaptive_config: 自适应检测配置（可选，使用默认值）
    :param topology_config: 拓扑拟合配置（可选，使用默认值）
    :param detector: 检测器实例（可复用）
    :return: AdaptiveDetectionResult
    
    示例：
        >>> from microfluidics_chip.core.config import Stage1Config, AdaptiveDetectionConfig
        >>> config = Stage1Config(yolo=..., geometry=...)
        >>> result = infer_stage1_adaptive("chip_001", image, config)
        >>> print(f"Detected: {len(result.detections)}, Dark: {result.dark_chamber_indices}")
    """
    start_time = time.time()
    
    logger.info(f"[{chip_id}] Starting adaptive Stage1 inference...")
    
    # ==================== 配置初始化 ====================
    if adaptive_config is None:
        adaptive_config = AdaptiveDetectionConfig()
        logger.info("Using default AdaptiveDetectionConfig")
    
    if topology_config is None:
        topology_config = TopologyConfig()
        logger.info("Using default TopologyConfig")
    
    # ==================== 检测器初始化 ====================
    if detector is None:
        detector = ChamberDetector(config.yolo)
        logger.info("Initialized new ChamberDetector")
    
    # ==================== Step 1: 自适应检测 ====================
    # 创建内部配置对象 (使用 dataclass 版本)
    internal_adaptive_config = AdaptiveConfig(
        coarse_imgsz=adaptive_config.coarse_imgsz,
        coarse_conf=adaptive_config.coarse_conf,
        fine_imgsz=adaptive_config.fine_imgsz,
        fine_conf=adaptive_config.fine_conf,
        cluster_eps=adaptive_config.cluster_eps,
        cluster_min_samples=adaptive_config.cluster_min_samples,
        roi_margin=adaptive_config.roi_margin,
        min_roi_size=adaptive_config.min_roi_size,
        enable_clahe=adaptive_config.enable_clahe,
        clahe_clip_limit=adaptive_config.clahe_clip_limit
    )
    
    adaptive_detector = AdaptiveDetector(internal_adaptive_config, detector)
    detections, cluster_result = adaptive_detector.detect_adaptive(raw_image)
    
    logger.info(f"[{chip_id}] Adaptive detection: {len(detections)} chambers, "
               f"cluster_score={cluster_result.cluster_score:.2f}")
    
    # ==================== Step 2: 拓扑拟合 ====================
    # 创建内部配置对象
    internal_topo_config = TopoConfig(
        template_scale=topology_config.template_scale,
        template_path=topology_config.template_path,
        ransac_iters=topology_config.ransac_iters,
        ransac_threshold=topology_config.ransac_threshold,
        min_inliers=topology_config.min_inliers,
        visibility_margin=topology_config.visibility_margin,
        brightness_roi_size=topology_config.brightness_roi_size,
        dark_percentile=topology_config.dark_percentile,
        fallback_to_affine=topology_config.fallback_to_affine
    )
    
    topology_fitter = TopologyFitter(internal_topo_config)
    
    # 提取检测中心点
    detected_centers = np.array([d.center for d in detections]) if detections else np.array([])
    
    fitting_result = topology_fitter.fit_and_fill(
        detected_centers=detected_centers,
        image_shape=raw_image.shape[:2],
        image=raw_image
    )
    
    logger.info(f"[{chip_id}] Topology fitting: success={fitting_result.fit_success}, "
               f"inlier_ratio={fitting_result.inlier_ratio:.2f}, "
               f"dark_chambers={fitting_result.dark_chamber_indices}")
    
    # ==================== 计算耗时 ====================
    processing_time = time.time() - start_time
    
    # ==================== 构造结果 ====================
    result = AdaptiveDetectionResult(
        detections=detections,
        roi_bbox=cluster_result.roi_bbox,
        cluster_score=cluster_result.cluster_score,
        is_fallback=cluster_result.is_fallback,
        fitted_centers=fitting_result.fitted_centers,
        visibility=fitting_result.visibility,
        detected_mask=fitting_result.detected_mask,
        dark_chamber_indices=fitting_result.dark_chamber_indices,
        inlier_ratio=fitting_result.inlier_ratio,
        reprojection_error=fitting_result.reprojection_error,
        fit_success=fitting_result.fit_success,
        processing_time=processing_time
    )
    
    logger.info(f"[{chip_id}] Adaptive Stage1 complete in {processing_time:.2f}s")
    
    return result
