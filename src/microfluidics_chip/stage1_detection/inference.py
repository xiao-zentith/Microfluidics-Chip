"""
Stage1 推理入口（纯算法逻辑，无IO）
遵循 v1.1 P3 强制规范：
- GT 处理时必须实例化独立的 geometry_engine
"""

import time
import numpy as np
from typing import Optional, Tuple
from ..core.types import Stage1Result
from ..core.config import Stage1Config
from ..core.logger import get_logger
from .detector import ChamberDetector
from .geometry_engine import CrossGeometryEngine

logger = get_logger("stage1_detection.inference")


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
    start_time = time.time()
    
    # ==================== 实例化（如果未提供） ====================
    if detector is None:
        detector = ChamberDetector(config.yolo)
        logger.info("Initialized new ChamberDetector")
    
    if geometry_engine is None:
        geometry_engine = CrossGeometryEngine(config.geometry)
        logger.info("Initialized new CrossGeometryEngine")
    
    # ==================== 检测 Raw 图像 ====================
    logger.info(f"[{chip_id}] Detecting chambers in raw image...")
    detections_raw = detector.detect(raw_image)
    
    if len(detections_raw) < 12:
        logger.error(f"[{chip_id}] Insufficient detections: {len(detections_raw)} < 12")
        raise ValueError(f"Chip {chip_id}: Only {len(detections_raw)} chambers detected")
    
    logger.info(f"[{chip_id}] Detected {len(detections_raw)} chambers")
    
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
        chambers=detections_raw,  # P0: List[ChamberDetection]
        gt_slices=gt_slices,
        debug_vis=debug_vis,
        processing_time=processing_time
    )
    
    logger.info(f"[{chip_id}] Stage1 inference complete in {processing_time:.2f}s")
    return result
