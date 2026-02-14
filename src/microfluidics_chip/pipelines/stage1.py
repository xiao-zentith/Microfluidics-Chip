"""
Stage1 业务编排层（IO + Inference）
遵循 v1.1 强制规范：
- P2: 固定文件命名
- P4: 批处理循环外初始化

v2.1 增强：
- 可选自适应粗到精检测链路
- 质量闸门 + 自动重试
- 失败时可回退标准检测流程
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from tqdm import tqdm
from ..core.types import Stage1Output, ChamberDetection, AdaptiveDetectionResult, Stage1Result
from ..core.config import (
    Stage1Config,
    AdaptiveDetectionConfig,
    TopologyConfig,
    AdaptiveRuntimeConfig
)
from ..core.io import save_stage1_result
from ..core.logger import get_logger
from ..stage1_detection.detector import ChamberDetector
from ..stage1_detection.geometry_engine import CrossGeometryEngine
from ..stage1_detection.preprocess import preprocess_image
from ..stage1_detection.inference import infer_stage1, infer_stage1_adaptive, infer_stage1_from_detections

logger = get_logger("pipelines.stage1")

ARM_TEMPLATE_INDICES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (9, 10, 11),
)
OUTERMOST_TEMPLATE_INDICES = (2, 5, 8, 11)


def _draw_yolo_detections(raw_image: np.ndarray, detections: List[ChamberDetection]) -> np.ndarray:
    """
    绘制 YOLO 原始检测结果（不做几何/拓扑后处理）。
    """
    vis = raw_image.copy()

    for det in detections:
        x, y, w, h = det.bbox
        x2, y2 = x + w, y + h
        color = (0, 255, 0) if det.class_id == 0 else (0, 165, 255)

        cv2.rectangle(vis, (x, y), (x2, y2), color, 2)
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)

        label = f"class={det.class_id} conf={det.confidence:.3f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y, th + baseline + 4)
        cv2.rectangle(vis, (x, ty - th - baseline - 4), (x + tw + 4, ty), color, -1)
        cv2.putText(
            vis,
            label,
            (x + 2, ty - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


def _build_unique_assignment(
    fitted_centers: np.ndarray,
    source_centers: np.ndarray,
    max_distance: float
) -> Dict[int, int]:
    """
    基于距离的贪心一对一匹配，避免同一检测被重复映射到多个模板点。
    """
    if len(fitted_centers) == 0 or len(source_centers) == 0:
        return {}

    dmat = np.linalg.norm(
        fitted_centers[:, None, :] - source_centers[None, :, :],
        axis=2
    )

    candidate_pairs: List[Tuple[float, int, int]] = []
    for fit_idx in range(dmat.shape[0]):
        for src_idx in range(dmat.shape[1]):
            dist = float(dmat[fit_idx, src_idx])
            if dist <= max_distance:
                candidate_pairs.append((dist, fit_idx, src_idx))

    candidate_pairs.sort(key=lambda x: x[0])

    assigned_fit = set()
    assigned_src = set()
    mapping: Dict[int, int] = {}

    for _, fit_idx, src_idx in candidate_pairs:
        if fit_idx in assigned_fit or src_idx in assigned_src:
            continue
        mapping[fit_idx] = src_idx
        assigned_fit.add(fit_idx)
        assigned_src.add(src_idx)

    return mapping


def _compute_arm_monotonicity(fitted_centers: np.ndarray) -> float:
    """
    计算四个旋臂是否满足“内->外距离单调递增”的比例。
    """
    if fitted_centers.shape[0] < 12:
        return 0.0

    centroid = np.mean(fitted_centers[:12], axis=0)
    pass_count = 0
    for arm in ARM_TEMPLATE_INDICES:
        dists = [
            float(np.linalg.norm(fitted_centers[idx] - centroid))
            for idx in arm
        ]
        if dists[0] < dists[1] < dists[2]:
            pass_count += 1

    return pass_count / len(ARM_TEMPLATE_INDICES)


def _resolve_runtime_config(
    config: Stage1Config,
    use_adaptive: Optional[bool]
) -> Tuple[bool, AdaptiveRuntimeConfig]:
    """
    合并 CLI 开关与配置文件的 adaptive runtime 设置。
    """
    runtime_config = (
        config.adaptive_runtime.model_copy(deep=True)
        if config.adaptive_runtime is not None
        else AdaptiveRuntimeConfig()
    )

    if use_adaptive is not None:
        runtime_config.enabled = use_adaptive

    return runtime_config.enabled, runtime_config


def _prepare_detection_image(
    raw_image: np.ndarray,
    preprocess_mode: str,
    clahe_clip_limit: float
) -> np.ndarray:
    """
    为本轮重试准备检测输入图像。
    """
    if preprocess_mode == "raw":
        return raw_image

    if preprocess_mode == "clahe":
        return preprocess_image(
            raw_image,
            enable_clahe=True,
            enable_invert=False,
            clahe_clip_limit=clahe_clip_limit
        )

    if preprocess_mode == "clahe_invert":
        return preprocess_image(
            raw_image,
            enable_clahe=True,
            enable_invert=True,
            clahe_clip_limit=clahe_clip_limit
        )

    # 未知模式时回退原图
    logger.warning(f"Unknown preprocess mode '{preprocess_mode}', fallback to raw")
    return raw_image


def _build_attempt_adaptive_config(
    base_config: AdaptiveDetectionConfig,
    runtime_config: AdaptiveRuntimeConfig,
    attempt_idx: int,
    preprocess_mode: str
) -> AdaptiveDetectionConfig:
    """
    基于基础配置构建某次重试的 adaptive 参数。
    """
    cfg = base_config.model_copy(deep=True)

    decay = runtime_config.confidence_decay ** attempt_idx
    cfg.coarse_conf = max(runtime_config.min_coarse_conf, cfg.coarse_conf * decay)
    cfg.fine_conf = max(runtime_config.min_fine_conf, cfg.fine_conf * decay)
    cfg.fine_imgsz = int(cfg.fine_imgsz + attempt_idx * runtime_config.fine_imgsz_step)

    # 如果本轮已做全图 CLAHE，就避免 ROI 阶段重复 CLAHE
    if preprocess_mode in ("clahe", "clahe_invert"):
        cfg.enable_clahe = False

    return cfg


def _build_geometry_detections(
    adaptive_result: AdaptiveDetectionResult,
    geometry_engine: CrossGeometryEngine,
    class_id_blank: int,
    class_id_lit: int,
    force_blank_if_missing: bool
) -> List[ChamberDetection]:
    """
    将拓扑拟合后的 12 个中心点转为几何引擎可消费的固定 12 检测结果。
    """
    fitted = np.asarray(adaptive_result.fitted_centers, dtype=np.float32)
    if fitted.ndim != 2 or fitted.shape[1] != 2:
        raise ValueError(f"Invalid fitted_centers shape: {fitted.shape}")

    if fitted.shape[0] < 12:
        raise ValueError(f"Insufficient fitted centers: {fitted.shape[0]} < 12")

    fitted = fitted[:12]
    source_dets = adaptive_result.detections
    source_centers = (
        np.array([d.center for d in source_dets], dtype=np.float32)
        if source_dets
        else np.empty((0, 2), dtype=np.float32)
    )

    patch_size = int(max(4, geometry_engine.config.crop_radius * 2))
    half = patch_size / 2.0
    detections: List[ChamberDetection] = []

    # 一对一匹配半径：随几何步长自适应，避免把远处干扰点“硬拉”到模板点
    match_radius = float(max(
        geometry_engine.config.ideal_chamber_step * 1.6,
        geometry_engine.config.crop_radius * 3
    ))
    match_map = _build_unique_assignment(fitted, source_centers, match_radius)

    for fit_idx, (cx, cy) in enumerate(fitted):
        class_id = int(class_id_lit)
        confidence = 0.0

        src_idx = match_map.get(fit_idx)
        if src_idx is not None:
            nearest = source_dets[src_idx]
            confidence = float(nearest.confidence)

        detections.append(
            ChamberDetection(
                bbox=(int(cx - half), int(cy - half), patch_size, patch_size),
                center=(float(cx), float(cy)),
                class_id=class_id,
                confidence=confidence
            )
        )

    # blank 身份不再依赖 YOLO 类别，优先使用拓扑暗腔室索引
    valid_dark = [
        int(idx) for idx in adaptive_result.dark_chamber_indices
        if 0 <= int(idx) < len(detections)
    ]
    blank_idx: Optional[int] = None

    if valid_dark:
        # 拓扑一般只给一个暗腔室，多个候选时优先使用最外侧索引
        outer_dark = [idx for idx in valid_dark if idx in OUTERMOST_TEMPLATE_INDICES]
        blank_idx = outer_dark[0] if outer_dark else valid_dark[0]
    elif force_blank_if_missing and detections:
        # 兜底：仅在最外侧四个位置里选置信度最低者，避免随机第一点
        blank_idx = min(
            OUTERMOST_TEMPLATE_INDICES,
            key=lambda idx: detections[idx].confidence if idx < len(detections) else float("inf")
        )

    if blank_idx is not None and 0 <= blank_idx < len(detections):
        normalized: List[ChamberDetection] = []
        for idx, det in enumerate(detections):
            normalized.append(
                ChamberDetection(
                    bbox=det.bbox,
                    center=det.center,
                    class_id=int(class_id_blank if idx == blank_idx else class_id_lit),
                    confidence=det.confidence
                )
            )
        detections = normalized

    return detections


def _collect_quality_metrics(
    adaptive_result: AdaptiveDetectionResult,
    geometry_detections: List[ChamberDetection],
    class_id_blank: int,
    preprocess_mode: str,
    attempt_idx: int,
    attempt_config: AdaptiveDetectionConfig
) -> Dict[str, Any]:
    """
    汇总质量闸门评估所需指标。
    """
    confidences_raw = [d.confidence for d in adaptive_result.detections]
    confidences_geo = [d.confidence for d in geometry_detections]
    mean_conf_raw = float(np.mean(confidences_raw)) if confidences_raw else 0.0
    mean_conf_geo = float(np.mean(confidences_geo)) if confidences_geo else 0.0

    fitted = np.asarray(adaptive_result.fitted_centers, dtype=np.float32)
    arm_monotonicity = _compute_arm_monotonicity(fitted) if fitted.ndim == 2 else 0.0

    blank_indices = [
        idx for idx, det in enumerate(geometry_detections)
        if det.class_id == class_id_blank
    ]
    blank_count = len(blank_indices)
    blank_on_outermost = (
        blank_count == 1 and blank_indices[0] in OUTERMOST_TEMPLATE_INDICES
    )

    metrics: Dict[str, Any] = {
        "attempt": attempt_idx + 1,
        "detection_count": len(adaptive_result.detections),
        "geometry_detection_count": len(geometry_detections),
        "fit_success": bool(adaptive_result.fit_success),
        "inlier_ratio": float(adaptive_result.inlier_ratio),
        "reprojection_error": float(adaptive_result.reprojection_error),
        "cluster_score": float(adaptive_result.cluster_score),
        "is_fallback_cluster": bool(adaptive_result.is_fallback),
        "mean_confidence": mean_conf_geo,
        "mean_confidence_raw": mean_conf_raw,
        "dark_chamber_count": len(adaptive_result.dark_chamber_indices),
        "blank_count": blank_count,
        "blank_indices": blank_indices,
        "blank_on_outermost": bool(blank_on_outermost),
        "arm_monotonicity": float(arm_monotonicity),
        "coarse_conf": float(attempt_config.coarse_conf),
        "fine_conf": float(attempt_config.fine_conf),
        "fine_imgsz": int(attempt_config.fine_imgsz),
    }
    # 字符串字段单独挂载，保持 metadata 可读
    metrics["preprocess_mode"] = preprocess_mode
    return metrics


def _quality_score(metrics: Dict[str, Any]) -> float:
    """
    计算综合质量分，用于多轮重试择优。
    """
    detection_term = min(metrics["detection_count"] / 12.0, 1.0)
    reproj_term = 1.0 - min(metrics["reprojection_error"], 100.0) / 100.0
    arm_term = float(metrics.get("arm_monotonicity", 0.0))
    blank_term = 1.0 if bool(metrics.get("blank_on_outermost", False)) else 0.0
    fit_bonus = 0.15 if metrics["fit_success"] else -0.15

    return (
        0.26 * metrics["inlier_ratio"]
        + 0.20 * metrics["cluster_score"]
        + 0.14 * metrics["mean_confidence"]
        + 0.12 * detection_term
        + 0.10 * reproj_term
        + 0.10 * arm_term
        + 0.08 * blank_term
        + fit_bonus
    )


def _passes_quality_gate(
    metrics: Dict[str, Any],
    runtime_config: AdaptiveRuntimeConfig
) -> bool:
    """
    质量闸门判断。
    """
    if runtime_config.require_fit_success and not metrics["fit_success"]:
        return False
    if metrics["detection_count"] < runtime_config.min_detections:
        return False
    if metrics["inlier_ratio"] < runtime_config.min_inlier_ratio:
        return False
    if metrics["reprojection_error"] > runtime_config.max_reprojection_error:
        return False
    if metrics["cluster_score"] < runtime_config.min_cluster_score:
        return False
    if metrics["mean_confidence"] < runtime_config.min_mean_confidence:
        return False
    if runtime_config.require_unique_blank and metrics.get("blank_count", 0) != 1:
        return False
    if runtime_config.require_blank_outermost and not metrics.get("blank_on_outermost", False):
        return False
    if metrics.get("arm_monotonicity", 0.0) < runtime_config.min_arm_monotonicity:
        return False
    return True


def _run_stage1_adaptive_with_retries(
    chip_id: str,
    raw_image: np.ndarray,
    gt_image: Optional[np.ndarray],
    config: Stage1Config,
    detector: ChamberDetector,
    geometry_engine: Optional[CrossGeometryEngine],
    runtime_config: AdaptiveRuntimeConfig
) -> Optional[Stage1Result]:
    """
    执行 adaptive 粗到精检测 + 质量闸门 + 自动重试。

    返回：
    - Stage1Result: adaptive 成功，或 best-effort（当 fallback_to_standard=False）
    - None: adaptive 未通过且允许回退到标准流程
    """
    adaptive_base = config.adaptive_detection or AdaptiveDetectionConfig()
    topology_config = config.topology or TopologyConfig()
    effective_geometry_engine = geometry_engine or CrossGeometryEngine(config.geometry)

    preprocess_sequence = runtime_config.preprocess_sequence or ["raw", "clahe", "clahe_invert"]
    max_attempts = max(1, runtime_config.max_attempts)

    best_payload = None
    best_score = float("-inf")

    for attempt_idx in range(max_attempts):
        preprocess_mode = preprocess_sequence[attempt_idx % len(preprocess_sequence)]
        attempt_config = _build_attempt_adaptive_config(
            adaptive_base, runtime_config, attempt_idx, preprocess_mode
        )
        try:
            detect_image = _prepare_detection_image(
                raw_image, preprocess_mode, attempt_config.clahe_clip_limit
            )

            adaptive_result = infer_stage1_adaptive(
                chip_id=chip_id,
                raw_image=detect_image,
                config=config,
                adaptive_config=attempt_config,
                topology_config=topology_config,
                detector=detector
            )

            geometry_detections = _build_geometry_detections(
                adaptive_result=adaptive_result,
                geometry_engine=effective_geometry_engine,
                class_id_blank=config.yolo.class_id_blank,
                class_id_lit=config.yolo.class_id_lit,
                force_blank_if_missing=runtime_config.force_blank_if_missing
            )

            metrics = _collect_quality_metrics(
                adaptive_result=adaptive_result,
                geometry_detections=geometry_detections,
                class_id_blank=config.yolo.class_id_blank,
                preprocess_mode=preprocess_mode,
                attempt_idx=attempt_idx,
                attempt_config=attempt_config
            )
            score = _quality_score(metrics)
            passed = _passes_quality_gate(metrics, runtime_config)
            metrics["quality_score"] = score
            metrics["quality_gate_passed"] = passed

            logger.info(
                f"[{chip_id}] Adaptive attempt {attempt_idx + 1}/{max_attempts}: "
                f"mode={preprocess_mode}, det={int(metrics['detection_count'])}, "
                f"inlier={metrics['inlier_ratio']:.2f}, reproj={metrics['reprojection_error']:.2f}, "
                f"blank_ok={metrics['blank_on_outermost']}, arm={metrics['arm_monotonicity']:.2f}, "
                f"score={score:.3f}, pass={passed}"
            )

            if score > best_score:
                best_score = score
                best_payload = (geometry_detections, metrics, attempt_idx)

            if passed:
                return infer_stage1_from_detections(
                    chip_id=chip_id,
                    raw_image=raw_image,
                    gt_image=gt_image,
                    detections_raw=geometry_detections,
                    config=config,
                    detector=detector,
                    geometry_engine=effective_geometry_engine,
                    quality_metrics=metrics,
                    quality_gate_passed=True,
                    detection_mode="adaptive",
                    retry_attempt=attempt_idx + 1
                )
        except Exception as e:
            logger.warning(
                f"[{chip_id}] Adaptive attempt {attempt_idx + 1}/{max_attempts} failed: {e}"
            )
            continue

    if best_payload is None:
        logger.warning(f"[{chip_id}] Adaptive retries produced no usable payload")
        return None

    if runtime_config.fallback_to_standard:
        logger.warning(
            f"[{chip_id}] Adaptive quality gate failed after {max_attempts} attempts, "
            "fallback to standard stage1 inference"
        )
        return None

    geometry_detections, metrics, attempt_idx = best_payload
    metrics["selected_by"] = "best_effort"
    logger.warning(
        f"[{chip_id}] Adaptive quality gate failed, using best-effort adaptive result "
        "(fallback_to_standard=False)"
    )
    return infer_stage1_from_detections(
        chip_id=chip_id,
        raw_image=raw_image,
        gt_image=gt_image,
        detections_raw=geometry_detections,
        config=config,
        detector=detector,
        geometry_engine=effective_geometry_engine,
        quality_metrics=metrics,
        quality_gate_passed=False,
        detection_mode="adaptive_best_effort",
        retry_attempt=attempt_idx + 1
    )


def run_stage1(
    chip_id: str,
    raw_image_path: Path,
    gt_image_path: Optional[Path],
    output_dir: Path,
    config: Stage1Config,
    detector: Optional[ChamberDetector] = None,
    geometry_engine: Optional[CrossGeometryEngine] = None,
    save_individual_slices: bool = False,
    save_debug: bool = True,
    use_adaptive: Optional[bool] = None
) -> Stage1Output:
    """
    运行单个芯片的 Stage1 处理
    
    流程：
    1. 读取图像
    2. 调用 infer_stage1() 推理
    3. P2: 使用 save_stage1_result() 保存（固定文件名）
    
    :param chip_id: 芯片ID
    :param raw_image_path: 原始图像路径
    :param gt_image_path: GT图像路径（可选）
    :param output_dir: 输出目录（会在此目录下创建 chip_id 子目录）
    :param config: Stage1 配置
    :param detector: 检测器实例（批处理时复用）
    :param geometry_engine: 几何引擎实例（批处理时复用）
    :param save_individual_slices: 是否保存单个切片图像（用于调试）
    :param save_debug: 是否保存调试可视化图像（检测框、类别标注等）
    :param use_adaptive: 是否启用 adaptive 推理（None 时跟随配置）
    :return: Stage1Output（落盘结果）
    """
    logger.info(f"[{chip_id}] Starting Stage1 processing...")
    
    # ==================== 读取图像 ====================
    raw_image = cv2.imread(str(raw_image_path))
    if raw_image is None:
        raise FileNotFoundError(f"Cannot read raw image: {raw_image_path}")
    
    gt_image = None
    if gt_image_path and gt_image_path.exists():
        gt_image = cv2.imread(str(gt_image_path))
        if gt_image is None:
            logger.warning(f"[{chip_id}] Cannot read GT image: {gt_image_path}")

    adaptive_enabled, runtime_config = _resolve_runtime_config(config, use_adaptive)

    # ==================== 推理 ====================
    if adaptive_enabled:
        logger.info(
            f"[{chip_id}] Adaptive detection enabled "
            f"(max_attempts={runtime_config.max_attempts})"
        )
        if detector is None:
            detector = ChamberDetector(config.yolo)

        adaptive_result = _run_stage1_adaptive_with_retries(
            chip_id=chip_id,
            raw_image=raw_image,
            gt_image=gt_image,
            config=config,
            detector=detector,
            geometry_engine=geometry_engine,
            runtime_config=runtime_config
        )

        if adaptive_result is not None:
            result = adaptive_result
        else:
            result = infer_stage1(
                chip_id=chip_id,
                raw_image=raw_image,
                gt_image=gt_image,
                config=config,
                detector=detector,
                geometry_engine=geometry_engine
            )
            result.detection_mode = "standard_fallback"
            result.quality_gate_passed = False
            result.retry_attempt = runtime_config.max_attempts
            result.quality_metrics = {
                "reason": "adaptive_quality_gate_failed",
                "adaptive_attempts": runtime_config.max_attempts
            }
    else:
        result = infer_stage1(
            chip_id=chip_id,
            raw_image=raw_image,
            gt_image=gt_image,
            config=config,
            detector=detector,
            geometry_engine=geometry_engine
        )
    
    # ==================== P2: 保存结果（固定文件名） ====================
    run_dir = output_dir / chip_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    save_gt = (result.gt_slices is not None)
    output = save_stage1_result(
        result, 
        run_dir, 
        save_gt=save_gt,
        save_individual_slices=save_individual_slices
    )
    
    # 保存调试可视化（如果有）
    if save_debug and result.debug_vis is not None:
        debug_path = run_dir / "debug_detection.png"
        cv2.imwrite(str(debug_path), result.debug_vis)
        logger.info(f"[{chip_id}] Debug visualization saved: debug_detection.png")
    
    logger.info(f"[{chip_id}] Stage1 output saved to: {run_dir}")
    logger.info(f"[{chip_id}] Files: stage1_metadata.json, aligned.png, chamber_slices.npz")
    
    return output


def run_stage1_yolo_only(
    chip_id: str,
    raw_image_path: Path,
    output_dir: Path,
    config: Stage1Config,
    detector: Optional[ChamberDetector] = None,
) -> Dict[str, Any]:
    """
    仅执行 YOLO 检测，不进行 Stage1 后处理（几何校正/拓扑拟合/切片）。

    输出文件:
    - raw.png
    - yolo_raw_detections.png
    - yolo_raw_detections.json
    """
    start = time.time()

    raw_image = cv2.imread(str(raw_image_path))
    if raw_image is None:
        raise FileNotFoundError(f"Cannot read raw image: {raw_image_path}")

    if detector is None:
        detector = ChamberDetector(config.yolo)

    detections = detector.detect(raw_image)
    vis = _draw_yolo_detections(raw_image, detections)

    run_dir = Path(output_dir) / chip_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_name = "raw.png"
    vis_name = "yolo_raw_detections.png"
    meta_name = "yolo_raw_detections.json"

    cv2.imwrite(str(run_dir / raw_name), raw_image)
    cv2.imwrite(str(run_dir / vis_name), vis)

    payload: Dict[str, Any] = {
        "chip_id": chip_id,
        "raw_image_path": raw_name,
        "visualization_path": vis_name,
        "num_detections": len(detections),
        "detections": [d.model_dump() for d in detections],
        "processing_time": time.time() - start,
    }

    with open(run_dir / meta_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[{chip_id}] YOLO-only inference saved: {run_dir / vis_name} "
        f"(detections={len(detections)})"
    )
    return payload


def run_stage1_yolo_only_batch(
    input_dir: Path,
    output_dir: Path,
    config: Stage1Config,
    image_extensions: List[str] = [".png", ".jpg", ".jpeg"],
    skip_suffix: str = "_gt",
) -> List[Dict[str, Any]]:
    """
    批量 YOLO-only 推理（不进行后处理）。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[Path] = []
    for ext in image_extensions:
        for p in input_dir.glob(f"*{ext}"):
            if skip_suffix and skip_suffix in p.stem:
                continue
            image_paths.append(p)

    if not image_paths:
        logger.warning(f"No image files found in {input_dir}")
        return []

    logger.info(f"Found {len(image_paths)} images for YOLO-only batch inference")
    detector = ChamberDetector(config.yolo)

    outputs: List[Dict[str, Any]] = []
    for p in tqdm(image_paths, desc="YOLO-only stage1"):
        chip_id = p.stem
        try:
            out = run_stage1_yolo_only(
                chip_id=chip_id,
                raw_image_path=p,
                output_dir=output_dir,
                config=config,
                detector=detector,
            )
            outputs.append(out)
        except Exception as e:
            logger.error(f"✗ {chip_id} failed in YOLO-only mode: {e}")

    logger.info(
        f"YOLO-only batch complete: {len(outputs)} success, {len(image_paths) - len(outputs)} failed"
    )
    return outputs


def run_stage1_batch(
    input_dir: Path,
    output_dir: Path,
    config: Stage1Config,
    gt_suffix: str = "_gt",
    image_extensions: List[str] = [".png", ".jpg", ".jpeg"],
    use_adaptive: Optional[bool] = None
) -> List[Stage1Output]:
    """
    批量运行 Stage1
    
    P4 强制规范：
    - 循环外初始化 detector 和 geometry_engine
    - 循环内复用实例（避免重复加载模型）
    
    目录结构假设：
    input_dir/
      ├── chip001.png
      ├── chip001_gt.png
      ├── chip002.png
      └── chip002_gt.png
    
    :param input_dir: 输入目录
    :param output_dir: 输出根目录
    :param config: Stage1 配置
    :param gt_suffix: GT 文件后缀（如 "_gt"）
    :param image_extensions: 支持的图像扩展名
    :param use_adaptive: 是否启用 adaptive 推理（None 时跟随配置）
    :return: Stage1Output 列表
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 收集文件对 ====================
    file_pairs = []
    
    for ext in image_extensions:
        for raw_path in input_dir.glob(f"*{ext}"):
            # 跳过 GT 文件
            if gt_suffix in raw_path.stem:
                continue
            
            chip_id = raw_path.stem
            
            # 查找对应的 GT 文件
            gt_path = raw_path.parent / f"{chip_id}{gt_suffix}{ext}"
            if not gt_path.exists():
                gt_path = None
            
            file_pairs.append((chip_id, raw_path, gt_path))
    
    if not file_pairs:
        logger.warning(f"No image files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(file_pairs)} image pairs")
    
    # ==================== P4: 循环外初始化模型 ====================
    logger.info("Initializing detector and geometry engine (P4 batch optimization)...")
    detector = ChamberDetector(config.yolo)
    geometry_engine = CrossGeometryEngine(config.geometry)
    logger.info("Models initialized successfully")
    
    # ==================== 批处理循环 ====================
    outputs = []
    
    for idx, (chip_id, raw_path, gt_path) in enumerate(tqdm(file_pairs, desc="Processing chips")):
        try:
            output = run_stage1(
                chip_id=chip_id,
                raw_image_path=raw_path,
                gt_image_path=gt_path,
                output_dir=output_dir,
                config=config,
                detector=detector,           # P4: 复用实例
                geometry_engine=geometry_engine,  # P4: 复用实例
                use_adaptive=use_adaptive
            )
            outputs.append(output)
            
            # 进度提示
            logger.info(f"✓ {chip_id} completed ({idx+1}/{len(file_pairs)})")
            
        except Exception as e:
            logger.error(f"✗ {chip_id} failed: {e}")
            continue
    
    # ==================== 汇总 ====================
    success_count = len(outputs)
    fail_count = len(file_pairs) - success_count
    
    logger.info(f"Batch processing complete: {success_count} success, {fail_count} failed")
    
    return outputs
