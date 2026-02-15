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
from ..stage1_detection.adaptive_detector import (
    AdaptiveDetector,
    AdaptiveDetectionConfig as InternalAdaptiveDetectionConfig
)
from ..stage1_detection.topology_fitter import (
    TopologyFitter,
    TopologyConfig as InternalTopologyConfig
)
from ..stage1_detection.preprocess import preprocess_image
from ..stage1_detection.inference import infer_stage1, infer_stage1_adaptive, infer_stage1_from_detections
from ..stage1_detection.topology_debug import debug_dump_topology

logger = get_logger("pipelines.stage1")

ARM_TEMPLATE_INDICES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (9, 10, 11),
)
OUTERMOST_TEMPLATE_INDICES = (2, 5, 8, 11)


class TopologyPostprocessError(RuntimeError):
    """
    后处理阶段异常，携带可落盘的 debug payload。
    """
    def __init__(self, message: str, debug_payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.debug_payload = debug_payload or {}


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


def _draw_adaptive_yolo_detections(
    raw_image: np.ndarray,
    detections: List[ChamberDetection],
    roi_bbox: Tuple[int, int, int, int],
    cluster_score: float,
    is_fallback: bool
) -> np.ndarray:
    """
    绘制自适应两阶段 YOLO 检测结果（仅检测，不含拓扑/几何后处理）。
    """
    vis = _draw_yolo_detections(raw_image, detections)
    x1, y1, x2, y2 = roi_bbox
    color = (0, 255, 255) if not is_fallback else (0, 165, 255)

    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    label = f"ROI score={cluster_score:.2f} fallback={int(is_fallback)}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    ty = max(y1, th + baseline + 6)
    cv2.rectangle(vis, (x1, ty - th - baseline - 6), (x1 + tw + 6, ty), color, -1)
    cv2.putText(
        vis,
        label,
        (x1 + 3, ty - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return vis


def _build_internal_adaptive_config(config: Stage1Config) -> InternalAdaptiveDetectionConfig:
    """
    将配置对象中的自适应参数转换为 AdaptiveDetector 所需 dataclass。
    """
    adaptive_config = config.adaptive_detection or AdaptiveDetectionConfig()
    return InternalAdaptiveDetectionConfig(
        coarse_imgsz=adaptive_config.coarse_imgsz,
        coarse_conf=adaptive_config.coarse_conf,
        fine_imgsz=adaptive_config.fine_imgsz,
        fine_conf=adaptive_config.fine_conf,
        cluster_eps=adaptive_config.cluster_eps,
        cluster_min_samples=adaptive_config.cluster_min_samples,
        roi_margin=adaptive_config.roi_margin,
        min_roi_size=adaptive_config.min_roi_size,
        enable_clahe=adaptive_config.enable_clahe,
        clahe_clip_limit=adaptive_config.clahe_clip_limit,
    )


def _build_internal_topology_config(config: Stage1Config) -> InternalTopologyConfig:
    """
    将配置对象中的拓扑参数转换为 TopologyFitter 所需 dataclass。
    """
    topology_config = config.topology or TopologyConfig()
    return InternalTopologyConfig(
        template_scale=topology_config.template_scale,
        template_path=topology_config.template_path,
        ransac_iters=topology_config.ransac_iters,
        ransac_threshold=topology_config.ransac_threshold,
        min_inliers=topology_config.min_inliers,
        visibility_margin=topology_config.visibility_margin,
        brightness_roi_size=topology_config.brightness_roi_size,
        dark_percentile=topology_config.dark_percentile,
        fallback_to_affine=topology_config.fallback_to_affine,
    )


def _compute_rotation_scale_from_matrix(matrix: np.ndarray) -> Tuple[float, float]:
    """
    从 2x3 仿射矩阵估计旋转角与缩放因子。
    """
    m = np.asarray(matrix, dtype=np.float32)
    if m.shape != (2, 3):
        return 0.0, 0.0

    sx = float(np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2))
    sy = float(np.sqrt(m[0, 1] ** 2 + m[1, 1] ** 2))
    scale = float((sx + sy) / 2.0)
    rotation = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
    return rotation, scale


def _compute_coverage_arms(detected_mask: np.ndarray) -> float:
    """
    计算模板四臂覆盖率（至少命中一个点的臂占比）。
    """
    if detected_mask.shape[0] < 12:
        return 0.0

    covered = 0
    for arm in ARM_TEMPLATE_INDICES:
        if any(bool(detected_mask[idx]) for idx in arm):
            covered += 1
    return covered / len(ARM_TEMPLATE_INDICES)


def _compute_candidate_dark_score(
    raw_image: np.ndarray,
    center: Tuple[float, float],
    roi_size: int
) -> Optional[float]:
    """
    使用圆形 ROI 上 V 通道鲁棒统计计算暗腔室分数（越低越暗）。
    """
    h, w = raw_image.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    half = max(4, int(roi_size // 2))

    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(w, int(cx + half))
    y2 = min(h, int(cy + half))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = raw_image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2].astype(np.float32)

    local_cx = cx - x1
    local_cy = cy - y1
    yy, xx = np.ogrid[:v_channel.shape[0], :v_channel.shape[1]]
    radius = max(3.0, min(v_channel.shape[0], v_channel.shape[1]) * 0.45)
    circle_mask = ((xx - local_cx) ** 2 + (yy - local_cy) ** 2) <= radius ** 2

    pixels = v_channel[circle_mask]
    if pixels.size < 20:
        pixels = v_channel.reshape(-1)
    if pixels.size == 0:
        return None

    p10 = float(np.percentile(pixels, 10))
    median = float(np.median(pixels))
    return 0.7 * p10 + 0.3 * median


def _select_blank_from_outermost(
    raw_image: np.ndarray,
    fitted_centers: np.ndarray,
    roi_size: int
) -> Tuple[List[int], Dict[int, float], float]:
    """
    在四个末端候选中按亮度最小规则选唯一 blank。
    """
    blank_scores: Dict[int, float] = {}
    for idx in OUTERMOST_TEMPLATE_INDICES:
        if idx >= fitted_centers.shape[0]:
            continue
        score = _compute_candidate_dark_score(
            raw_image=raw_image,
            center=(float(fitted_centers[idx, 0]), float(fitted_centers[idx, 1])),
            roi_size=roi_size
        )
        if score is not None:
            blank_scores[idx] = float(score)

    if not blank_scores:
        return [], {}, 0.0

    blank_idx = min(blank_scores, key=blank_scores.get)
    sorted_scores = sorted(blank_scores.values())
    if len(sorted_scores) > 1:
        gap = sorted_scores[1] - sorted_scores[0]
        blank_confidence = float(np.clip(gap / (sorted_scores[1] + 1e-6), 0.0, 1.0))
    else:
        blank_confidence = 0.0

    return [int(blank_idx)], blank_scores, blank_confidence


def _build_relaxed_post_adaptive_config(config: Stage1Config) -> AdaptiveDetectionConfig:
    """
    构建后处理失败时的宽松重检参数。
    """
    base = (config.adaptive_detection or AdaptiveDetectionConfig()).model_copy(deep=True)
    base.coarse_conf = max(0.02, base.coarse_conf * 0.60)
    base.fine_conf = max(0.08, base.fine_conf * 0.60)
    base.fine_imgsz = int(base.fine_imgsz + 256)
    base.enable_clahe = True
    return base


def _run_topology_refine_postprocess(
    chip_id: str,
    raw_image: np.ndarray,
    source_detections: List[ChamberDetection],
    config: Stage1Config,
    min_topology_detections: int,
    source_tag: str,
    raw_image_path: Path,
    attempt_id: str,
    used_fallback: bool,
    roi_bbox: Optional[Any] = None,
    detection_params: Optional[Dict[str, Any]] = None,
    geometry_engine: Optional[CrossGeometryEngine] = None
) -> Tuple[List[ChamberDetection], Dict[str, Any], Dict[str, Any]]:
    """
    对检测点执行拓扑拟合回填，输出固定12点几何检测及 QC。
    """
    n_det = len(source_detections)
    topology_cfg = _build_internal_topology_config(config)
    fill_match_thresh_px = float(max(15.0, config.geometry.crop_radius * 0.6))

    if n_det < min_topology_detections:
        reason = (
            "postprocess_qc_failed: insufficient_detections_for_topology "
            f"(n_det={n_det}, min_required={min_topology_detections})"
        )
        debug_payload = _build_topology_attempt_debug_payload(
            chip_id=chip_id,
            attempt_id=attempt_id,
            status="failed",
            raw_image_path=raw_image_path,
            source_tag=source_tag,
            used_fallback=used_fallback,
            min_topology_detections=min_topology_detections,
            source_detections=source_detections,
            topology_cfg=topology_cfg,
            fit_success=False,
            transform_type="none",
            n_inliers=0,
            rmse_px=None,
            scale=None,
            rotation_deg=None,
            coverage_arms=0.0,
            inlier_indices=[],
            fitted_centers=None,
            visibility=None,
            detected_mask=None,
            blank_scores={},
            blank_indices=[],
            blank_confidence=0.0,
            fill_match_thresh_px=fill_match_thresh_px,
            roi_bbox=roi_bbox,
            failure_reason=reason,
            detection_params=detection_params,
            qc={
                "source": source_tag,
                "n_det": int(n_det),
                "fit_success": False,
                "fit_success_raw": False,
            },
        )
        raise TopologyPostprocessError(reason, debug_payload=debug_payload)

    fitter = TopologyFitter(topology_cfg)

    detected_centers = np.array([d.center for d in source_detections], dtype=np.float32)
    fitting_result = fitter.fit_and_fill(
        detected_centers=detected_centers,
        image_shape=raw_image.shape[:2],
        image=None
    )

    rotation_deg, scale = _compute_rotation_scale_from_matrix(fitting_result.transform_matrix)
    coverage_arms = _compute_coverage_arms(fitting_result.detected_mask)
    n_inliers = int(len(fitting_result.inlier_indices))
    rmse_px = float(fitting_result.reprojection_error)
    blank_indices, blank_scores, blank_confidence = _select_blank_from_outermost(
        raw_image=raw_image,
        fitted_centers=np.asarray(fitting_result.fitted_centers, dtype=np.float32),
        roi_size=topology_cfg.brightness_roi_size
    )
    fit_success_qc = bool(
        fitting_result.fit_success
        and n_inliers >= int(topology_cfg.min_inliers)
        and np.isfinite(rmse_px)
        and coverage_arms >= 0.25
        and scale > 0.0
    )

    qc: Dict[str, Any] = {
        "source": source_tag,
        "n_det": int(n_det),
        "n_inliers": int(n_inliers),
        "n_filled": int(max(0, 12 - int(np.sum(fitting_result.detected_mask)))),
        "rmse_px": float(rmse_px),
        "inlier_ratio": float(fitting_result.inlier_ratio),
        "coverage_arms": float(coverage_arms),
        "scale": float(scale),
        "rotation": float(rotation_deg),
        "transform_type": str(fitting_result.transform_type),
        "blank_scores": {str(k): float(v) for k, v in blank_scores.items()},
        "blank_confidence": float(blank_confidence),
        "blank_indices": [int(x) for x in blank_indices],
        "fit_success_raw": bool(fitting_result.fit_success),
        "fit_success": bool(fit_success_qc),
    }

    debug_payload = _build_topology_attempt_debug_payload(
        chip_id=chip_id,
        attempt_id=attempt_id,
        status="success",
        raw_image_path=raw_image_path,
        source_tag=source_tag,
        used_fallback=used_fallback,
        min_topology_detections=min_topology_detections,
        source_detections=source_detections,
        topology_cfg=topology_cfg,
        fit_success=fit_success_qc,
        transform_type=str(fitting_result.transform_type),
        n_inliers=n_inliers,
        rmse_px=rmse_px,
        scale=scale,
        rotation_deg=rotation_deg,
        coverage_arms=coverage_arms,
        inlier_indices=[int(i) for i in fitting_result.inlier_indices],
        fitted_centers=np.asarray(fitting_result.fitted_centers, dtype=np.float32),
        visibility=np.asarray(fitting_result.visibility, dtype=bool),
        detected_mask=np.asarray(fitting_result.detected_mask, dtype=bool),
        blank_scores=blank_scores,
        blank_indices=blank_indices,
        blank_confidence=blank_confidence,
        fill_match_thresh_px=fill_match_thresh_px,
        roi_bbox=roi_bbox,
        failure_reason=None,
        detection_params=detection_params,
        qc=qc,
    )

    if not fit_success_qc:
        reason = (
            "postprocess_qc_failed: topology_fit_failed "
            f"(n_det={n_det}, n_inliers={n_inliers}, rmse_px={rmse_px:.2f}, coverage_arms={coverage_arms:.2f})"
        )
        debug_payload["status"] = "failed"
        debug_payload["failure_reason"] = reason
        raise TopologyPostprocessError(reason, debug_payload=debug_payload)
    if not blank_indices:
        reason = (
            "postprocess_qc_failed: blank_unresolved "
            f"(n_det={n_det}, coverage_arms={coverage_arms:.2f})"
        )
        debug_payload["status"] = "failed"
        debug_payload["failure_reason"] = reason
        raise TopologyPostprocessError(reason, debug_payload=debug_payload)

    adaptive_result = AdaptiveDetectionResult(
        detections=source_detections,
        roi_bbox=(0, 0, int(raw_image.shape[1]), int(raw_image.shape[0])),
        cluster_score=0.0,
        is_fallback=(source_tag != "json"),
        fitted_centers=np.asarray(fitting_result.fitted_centers, dtype=np.float32),
        visibility=np.asarray(fitting_result.visibility, dtype=bool),
        detected_mask=np.asarray(fitting_result.detected_mask, dtype=bool),
        dark_chamber_indices=blank_indices,
        inlier_ratio=float(fitting_result.inlier_ratio),
        reprojection_error=float(fitting_result.reprojection_error),
        fit_success=bool(fitting_result.fit_success),
        processing_time=0.0
    )

    effective_geometry_engine = geometry_engine or CrossGeometryEngine(config.geometry)
    geometry_detections = _build_geometry_detections(
        adaptive_result=adaptive_result,
        geometry_engine=effective_geometry_engine,
        class_id_blank=config.yolo.class_id_blank,
        class_id_lit=config.yolo.class_id_lit,
        force_blank_if_missing=False
    )
    return geometry_detections, qc, debug_payload


def _load_detection_payload(detections_json_path: Path) -> Dict[str, Any]:
    """
    读取检测结果 JSON。
    """
    path = Path(detections_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Detection json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "detections" not in payload:
        raise ValueError(f"Detection json missing 'detections': {path}")
    return payload


def _parse_detections_from_payload(
    payload: Dict[str, Any],
    source_path: Path
) -> List[ChamberDetection]:
    """
    将 JSON payload 解析为 ChamberDetection 列表。
    """
    raw_detections = payload.get("detections")
    if not isinstance(raw_detections, list):
        raise ValueError(f"Invalid detections format in {source_path}")

    detections: List[ChamberDetection] = []
    for idx, item in enumerate(raw_detections):
        try:
            detections.append(ChamberDetection.model_validate(item))
        except Exception as e:
            raise ValueError(
                f"Invalid detection at index {idx} in {source_path}: {e}"
            ) from e
    return detections


def _resolve_raw_image_path(
    detections_json_path: Path,
    payload: Dict[str, Any],
    raw_image_override: Optional[Path]
) -> Path:
    """
    解析后处理要使用的原图路径。
    """
    if raw_image_override is not None:
        candidate = Path(raw_image_override)
        if not candidate.exists():
            raise FileNotFoundError(f"Raw image not found: {candidate}")
        return candidate

    raw_rel = payload.get("raw_image_path")
    candidates: List[Path] = []
    if isinstance(raw_rel, str) and raw_rel:
        candidates.append(detections_json_path.parent / raw_rel)
    candidates.append(detections_json_path.parent / "raw.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve raw image for {detections_json_path}. "
        f"Tried: {[str(p) for p in candidates]}"
    )

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


def _serialize_detections_for_debug(detections: List[ChamberDetection]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for idx, det in enumerate(detections):
        payload.append(
            {
                "det_idx": int(idx),
                "x": float(det.center[0]),
                "y": float(det.center[1]),
                "conf": float(det.confidence),
                "class_id": int(det.class_id),
                "bbox": [int(det.bbox[0]), int(det.bbox[1]), int(det.bbox[2]), int(det.bbox[3])],
            }
        )
    return payload


def _normalize_roi_bbox(roi_bbox: Optional[Any]) -> Optional[List[int]]:
    if roi_bbox is None:
        return None
    if not isinstance(roi_bbox, (list, tuple)) or len(roi_bbox) != 4:
        return None
    try:
        return [int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3])]
    except Exception:
        return None


def _build_fitted_point_debug(
    fitted_centers: np.ndarray,
    source_detections: List[ChamberDetection],
    visibility: Optional[np.ndarray],
    detected_mask: Optional[np.ndarray],
    fill_match_thresh_px: float
) -> Dict[str, Any]:
    src_centers = (
        np.array([d.center for d in source_detections], dtype=np.float32)
        if source_detections
        else np.empty((0, 2), dtype=np.float32)
    )

    fitted_points: List[Dict[str, Any]] = []
    pure_filled_count = 0

    for idx in range(int(fitted_centers.shape[0])):
        cx = float(fitted_centers[idx, 0])
        cy = float(fitted_centers[idx, 1])

        if src_centers.shape[0] > 0:
            dists = np.linalg.norm(src_centers - np.array([cx, cy], dtype=np.float32), axis=1)
            matched_det_idx = int(np.argmin(dists))
            match_dist = float(np.min(dists))
        else:
            matched_det_idx = None
            match_dist = None

        pure_filled = bool(match_dist is None or match_dist > fill_match_thresh_px)
        if pure_filled:
            pure_filled_count += 1

        fitted_points.append(
            {
                "template_index": int(idx),
                "x": cx,
                "y": cy,
                "visibility": bool(visibility[idx]) if visibility is not None and idx < len(visibility) else None,
                "detected_by_model": bool(detected_mask[idx]) if detected_mask is not None and idx < len(detected_mask) else None,
                "matched_det_idx": matched_det_idx,
                "match_dist_px": match_dist,
                "pure_filled": pure_filled,
            }
        )

    ratio = float(pure_filled_count / max(1, len(fitted_points)))
    return {
        "fitted_points": fitted_points,
        "pure_filled_count": int(pure_filled_count),
        "pure_filled_ratio": ratio,
    }


def _build_topology_attempt_debug_payload(
    *,
    chip_id: str,
    attempt_id: str,
    status: str,
    raw_image_path: Path,
    source_tag: str,
    used_fallback: bool,
    min_topology_detections: int,
    source_detections: List[ChamberDetection],
    topology_cfg: InternalTopologyConfig,
    fit_success: bool = False,
    transform_type: str = "none",
    n_inliers: int = 0,
    rmse_px: Optional[float] = None,
    scale: Optional[float] = None,
    rotation_deg: Optional[float] = None,
    coverage_arms: Optional[float] = None,
    inlier_indices: Optional[List[int]] = None,
    fitted_centers: Optional[np.ndarray] = None,
    visibility: Optional[np.ndarray] = None,
    detected_mask: Optional[np.ndarray] = None,
    blank_scores: Optional[Dict[int, float]] = None,
    blank_indices: Optional[List[int]] = None,
    blank_confidence: Optional[float] = None,
    fill_match_thresh_px: float = 15.0,
    roi_bbox: Optional[Any] = None,
    failure_reason: Optional[str] = None,
    detection_params: Optional[Dict[str, Any]] = None,
    qc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    detections_payload = _serialize_detections_for_debug(source_detections)

    fit_block: Dict[str, Any] = {
        "fit_success": bool(fit_success),
        "transform_type": str(transform_type),
        "n_inliers": int(n_inliers),
        "rmse_px": float(rmse_px) if rmse_px is not None and np.isfinite(rmse_px) else None,
        "scale": float(scale) if scale is not None and np.isfinite(scale) else None,
        "rotation_deg": float(rotation_deg) if rotation_deg is not None and np.isfinite(rotation_deg) else None,
        "coverage_arms": float(coverage_arms) if coverage_arms is not None and np.isfinite(coverage_arms) else None,
        "fill_match_thresh_px": float(fill_match_thresh_px),
    }

    template_name = (
        Path(topology_cfg.template_path).name
        if topology_cfg.template_path
        else "default_cross_template"
    )
    template_block = {
        "template_name": template_name,
        "template_version": "cross12_v1",
        "template_scale": float(topology_cfg.template_scale),
        "template_path": topology_cfg.template_path,
    }

    fitted_block = {
        "fitted_points": [],
        "pure_filled_count": 0,
        "pure_filled_ratio": 0.0,
    }
    if fitted_centers is not None and isinstance(fitted_centers, np.ndarray) and fitted_centers.ndim == 2:
        fitted_block = _build_fitted_point_debug(
            fitted_centers=fitted_centers,
            source_detections=source_detections,
            visibility=visibility,
            detected_mask=detected_mask,
            fill_match_thresh_px=fill_match_thresh_px
        )

    blank_scores_dict = {str(int(k)): float(v) for k, v in (blank_scores or {}).items()}
    blank_indices = [int(i) for i in (blank_indices or [])]
    blank_id = int(blank_indices[0]) if blank_indices else None

    return {
        "chip_id": chip_id,
        "attempt_id": attempt_id,
        "status": status,
        "failure_reason": failure_reason,
        "raw_image_path": str(raw_image_path),
        "coordinate_frame": "raw_image",
        "roi_bbox": _normalize_roi_bbox(roi_bbox),
        "used_fallback": bool(used_fallback),
        "source": source_tag,
        "input": {
            "min_topology_detections": int(min_topology_detections),
            "detection_params": detection_params or {},
        },
        "n_det": int(len(source_detections)),
        "detections": detections_payload,
        "inlier_det_indices": [int(i) for i in (inlier_indices or [])],
        "fit": fit_block,
        "template": template_block,
        "fitted_points": fitted_block["fitted_points"],
        "pure_filled_count": int(fitted_block["pure_filled_count"]),
        "pure_filled_ratio": float(fitted_block["pure_filled_ratio"]),
        "blank": {
            "candidate_indices": [int(i) for i in OUTERMOST_TEMPLATE_INDICES],
            "blank_scores": blank_scores_dict,
            "blank_id": blank_id,
            "blank_confidence": float(blank_confidence) if blank_confidence is not None else None,
            "score_method": "0.7*p10(V)+0.3*median(V), circle_roi",
        },
        "qc": qc or {},
    }


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


def run_stage1_yolo_adaptive_only(
    chip_id: str,
    raw_image_path: Path,
    output_dir: Path,
    config: Stage1Config,
    detector: Optional[ChamberDetector] = None,
) -> Dict[str, Any]:
    """
    仅执行“两阶段 YOLO 检测”：
    global coarse scan -> cluster ROI -> fine scan。
    不执行拓扑拟合与几何后处理。

    输出文件:
    - raw.png
    - adaptive_yolo_raw_detections.png
    - adaptive_yolo_raw_detections.json
    """
    start = time.time()

    raw_image = cv2.imread(str(raw_image_path))
    if raw_image is None:
        raise FileNotFoundError(f"Cannot read raw image: {raw_image_path}")

    if detector is None:
        detector = ChamberDetector(config.yolo)

    adaptive_detector = AdaptiveDetector(_build_internal_adaptive_config(config), detector)
    detections, cluster_result = adaptive_detector.detect_adaptive(raw_image)

    vis = _draw_adaptive_yolo_detections(
        raw_image=raw_image,
        detections=detections,
        roi_bbox=cluster_result.roi_bbox,
        cluster_score=cluster_result.cluster_score,
        is_fallback=cluster_result.is_fallback,
    )

    run_dir = Path(output_dir) / chip_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_name = "raw.png"
    vis_name = "adaptive_yolo_raw_detections.png"
    meta_name = "adaptive_yolo_raw_detections.json"

    cv2.imwrite(str(run_dir / raw_name), raw_image)
    cv2.imwrite(str(run_dir / vis_name), vis)

    adaptive_cfg = config.adaptive_detection or AdaptiveDetectionConfig()
    payload: Dict[str, Any] = {
        "chip_id": chip_id,
        "raw_image_path": raw_name,
        "visualization_path": vis_name,
        "num_detections": len(detections),
        "detections": [d.model_dump() for d in detections],
        "roi_bbox": list(cluster_result.roi_bbox),
        "cluster_score": float(cluster_result.cluster_score),
        "is_fallback": bool(cluster_result.is_fallback),
        "num_clusters_found": int(cluster_result.num_clusters_found),
        "cluster_centers": [list(c) for c in cluster_result.cluster_centers],
        "adaptive_config": adaptive_cfg.model_dump(),
        "processing_time": time.time() - start,
    }

    with open(run_dir / meta_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[{chip_id}] Adaptive YOLO-only inference saved: {run_dir / vis_name} "
        f"(detections={len(detections)}, cluster_score={cluster_result.cluster_score:.2f})"
    )
    return payload


def run_stage1_yolo_adaptive_only_batch(
    input_dir: Path,
    output_dir: Path,
    config: Stage1Config,
    image_extensions: List[str] = [".png", ".jpg", ".jpeg"],
    skip_suffix: str = "_gt",
) -> List[Dict[str, Any]]:
    """
    批量“两阶段 YOLO 仅检测”推理（不进行拓扑/几何后处理）。
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

    logger.info(f"Found {len(image_paths)} images for adaptive YOLO-only batch inference")
    detector = ChamberDetector(config.yolo)

    outputs: List[Dict[str, Any]] = []
    for p in tqdm(image_paths, desc="Adaptive YOLO-only stage1"):
        chip_id = p.stem
        try:
            out = run_stage1_yolo_adaptive_only(
                chip_id=chip_id,
                raw_image_path=p,
                output_dir=output_dir,
                config=config,
                detector=detector,
            )
            outputs.append(out)
        except Exception as e:
            logger.error(f"✗ {chip_id} failed in adaptive YOLO-only mode: {e}")

    logger.info(
        "Adaptive YOLO-only batch complete: "
        f"{len(outputs)} success, {len(image_paths) - len(outputs)} failed"
    )
    return outputs


def run_stage1_postprocess_from_json(
    detections_json_path: Path,
    output_dir: Path,
    config: Stage1Config,
    raw_image_path: Optional[Path] = None,
    chip_id: Optional[str] = None,
    min_topology_detections: Optional[int] = None,
    enable_fallback_detection: bool = True,
    geometry_engine: Optional[CrossGeometryEngine] = None,
    save_individual_slices: bool = False,
    save_debug: bool = True
) -> Stage1Output:
    """
    仅执行 Stage1 后处理（几何校正 + 切片），检测结果从 JSON 读取。

    支持输入：
    - stage1-yolo 的 yolo_raw_detections.json
    - stage1-yolo-adaptive 的 adaptive_yolo_raw_detections.json
    """
    detections_json_path = Path(detections_json_path)
    payload = _load_detection_payload(detections_json_path)
    detections_from_json = _parse_detections_from_payload(payload, detections_json_path)
    resolved_raw_image = _resolve_raw_image_path(
        detections_json_path=detections_json_path,
        payload=payload,
        raw_image_override=raw_image_path
    )

    final_chip_id = chip_id or payload.get("chip_id") or resolved_raw_image.stem
    run_dir = Path(output_dir) / final_chip_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_image = cv2.imread(str(resolved_raw_image))
    if raw_image is None:
        raise FileNotFoundError(f"Cannot read raw image: {resolved_raw_image}")

    effective_geometry_engine = geometry_engine or CrossGeometryEngine(config.geometry)
    runtime_cfg = config.adaptive_runtime or AdaptiveRuntimeConfig()
    min_required = int(
        min_topology_detections
        if min_topology_detections is not None
        else runtime_cfg.min_detections
    )
    min_required = max(2, min(12, min_required))

    attempts: List[Dict[str, Any]] = []
    selected_attempt_payload: Optional[Dict[str, Any]] = None
    selected_geometry_detections: Optional[List[ChamberDetection]] = None
    selected_qc: Dict[str, Any] = {}
    selected_source = "json"

    json_detection_params: Dict[str, Any] = {
        "source_json": str(detections_json_path),
        "adaptive_config_from_json": payload.get("adaptive_config"),
        "preprocess_mode": "from_json",
    }
    json_roi_bbox = payload.get("roi_bbox")

    # ---------- Try 1: detections from JSON ----------
    try:
        geometry_detections, qc, attempt_payload = _run_topology_refine_postprocess(
            chip_id=final_chip_id,
            raw_image=raw_image,
            source_detections=detections_from_json,
            config=config,
            min_topology_detections=min_required,
            source_tag="json",
            raw_image_path=resolved_raw_image,
            attempt_id="try1_json",
            used_fallback=False,
            roi_bbox=json_roi_bbox,
            detection_params=json_detection_params,
            geometry_engine=effective_geometry_engine
        )
        debug_dump_topology(run_dir, raw_image, attempt_payload, suffix="_try1")
        attempts.append(attempt_payload)
        selected_attempt_payload = attempt_payload
        selected_geometry_detections = geometry_detections
        selected_qc = qc
    except TopologyPostprocessError as first_err:
        first_payload = first_err.debug_payload or _build_topology_attempt_debug_payload(
            chip_id=final_chip_id,
            attempt_id="try1_json",
            status="failed",
            raw_image_path=resolved_raw_image,
            source_tag="json",
            used_fallback=False,
            min_topology_detections=min_required,
            source_detections=detections_from_json,
            topology_cfg=_build_internal_topology_config(config),
            failure_reason=str(first_err),
            detection_params=json_detection_params,
            roi_bbox=json_roi_bbox,
            fit_success=False,
            transform_type="none",
            n_inliers=0,
            qc={"source": "json", "n_det": len(detections_from_json), "fit_success": False},
        )
        debug_dump_topology(run_dir, raw_image, first_payload, suffix="_try1")
        attempts.append(first_payload)

        if not enable_fallback_detection:
            final_payload = dict(first_payload)
            final_payload["used_fallback"] = False
            final_payload["final_status"] = "failed"
            final_payload["final_reason"] = str(first_err)
            final_payload["attempts"] = attempts
            final_payload["selected_attempt_id"] = "try1_json"
            debug_dump_topology(run_dir, raw_image, final_payload, suffix=None)
            raise RuntimeError(str(first_err)) from first_err

        # ---------- Try 2: relaxed fallback adaptive detection ----------
        logger.warning(
            f"[{final_chip_id}] postprocess topology refine failed from json detections: {first_err}. "
            "Retrying with relaxed adaptive detection."
        )
        try:
            detector = ChamberDetector(config.yolo)
        except Exception as detector_err:
            if str(config.yolo.device).startswith("cuda"):
                logger.warning(
                    f"[{final_chip_id}] fallback detector init on cuda failed ({detector_err}), retry on cpu."
                )
                cpu_yolo_cfg = config.yolo.model_copy(deep=True)
                cpu_yolo_cfg.device = "cpu"
                detector = ChamberDetector(cpu_yolo_cfg)
            else:
                raise

        relaxed_cfg = _build_relaxed_post_adaptive_config(config)
        adaptive_detector = AdaptiveDetector(
            InternalAdaptiveDetectionConfig(
                coarse_imgsz=relaxed_cfg.coarse_imgsz,
                coarse_conf=relaxed_cfg.coarse_conf,
                fine_imgsz=relaxed_cfg.fine_imgsz,
                fine_conf=relaxed_cfg.fine_conf,
                cluster_eps=relaxed_cfg.cluster_eps,
                cluster_min_samples=relaxed_cfg.cluster_min_samples,
                roi_margin=relaxed_cfg.roi_margin,
                min_roi_size=relaxed_cfg.min_roi_size,
                enable_clahe=relaxed_cfg.enable_clahe,
                clahe_clip_limit=relaxed_cfg.clahe_clip_limit,
            ),
            detector
        )
        fallback_detections, fallback_cluster = adaptive_detector.detect_adaptive(raw_image)
        fallback_detection_params = {
            "preprocess_mode": "fallback_adaptive",
            "coarse_conf": float(relaxed_cfg.coarse_conf),
            "fine_conf": float(relaxed_cfg.fine_conf),
            "fine_imgsz": int(relaxed_cfg.fine_imgsz),
            "enable_clahe": bool(relaxed_cfg.enable_clahe),
            "clahe_clip_limit": float(relaxed_cfg.clahe_clip_limit),
        }

        try:
            geometry_detections, qc, second_payload = _run_topology_refine_postprocess(
                chip_id=final_chip_id,
                raw_image=raw_image,
                source_detections=fallback_detections,
                config=config,
                min_topology_detections=min_required,
                source_tag="fallback_adaptive",
                raw_image_path=resolved_raw_image,
                attempt_id="try2_fallback",
                used_fallback=True,
                roi_bbox=fallback_cluster.roi_bbox,
                detection_params=fallback_detection_params,
                geometry_engine=effective_geometry_engine
            )
            second_payload["fallback_reason"] = str(first_err)
            debug_dump_topology(run_dir, raw_image, second_payload, suffix="_try2")
            attempts.append(second_payload)

            selected_attempt_payload = second_payload
            selected_geometry_detections = geometry_detections
            selected_qc = qc
            selected_source = "fallback_adaptive"
        except TopologyPostprocessError as second_err:
            second_payload = second_err.debug_payload or _build_topology_attempt_debug_payload(
                chip_id=final_chip_id,
                attempt_id="try2_fallback",
                status="failed",
                raw_image_path=resolved_raw_image,
                source_tag="fallback_adaptive",
                used_fallback=True,
                min_topology_detections=min_required,
                source_detections=fallback_detections,
                topology_cfg=_build_internal_topology_config(config),
                failure_reason=str(second_err),
                detection_params=fallback_detection_params,
                roi_bbox=fallback_cluster.roi_bbox,
                fit_success=False,
                transform_type="none",
                n_inliers=0,
                qc={"source": "fallback_adaptive", "n_det": len(fallback_detections), "fit_success": False},
            )
            second_payload["fallback_reason"] = str(first_err)
            debug_dump_topology(run_dir, raw_image, second_payload, suffix="_try2")
            attempts.append(second_payload)

            final_payload = dict(second_payload)
            final_payload["used_fallback"] = True
            final_payload["final_status"] = "failed"
            final_payload["final_reason"] = str(second_err)
            final_payload["attempts"] = attempts
            final_payload["selected_attempt_id"] = "try2_fallback"
            debug_dump_topology(run_dir, raw_image, final_payload, suffix=None)
            raise RuntimeError(str(second_err)) from second_err

    if selected_geometry_detections is None or selected_attempt_payload is None:
        raise RuntimeError(f"[{final_chip_id}] unexpected empty postprocess output")

    selected_qc.update(
        {
            "source_detections_json": str(detections_json_path),
            "input_detection_count": int(len(detections_from_json)),
            "selected_detection_count": int(len(selected_geometry_detections)),
            "min_topology_detections": int(min_required),
        }
    )
    if selected_source != "json":
        selected_qc["fallback_reason"] = attempts[0].get("failure_reason") if attempts else None

    logger.info(
        f"[{final_chip_id}] postprocess QC: source={selected_qc.get('source')}, "
        f"n_det={selected_qc.get('n_det')}, n_inliers={selected_qc.get('n_inliers')}, rmse_px={selected_qc.get('rmse_px'):.2f}, "
        f"coverage_arms={selected_qc.get('coverage_arms'):.2f}, scale={selected_qc.get('scale'):.3f}, "
        f"rotation={selected_qc.get('rotation'):.2f}, blank_scores={selected_qc.get('blank_scores')}, "
        f"blank_confidence={selected_qc.get('blank_confidence'):.3f}"
    )

    # Final merged debug (required fixed filename)
    final_debug_payload = dict(selected_attempt_payload)
    final_debug_payload["attempts"] = attempts
    final_debug_payload["used_fallback"] = (selected_source != "json")
    final_debug_payload["selected_attempt_id"] = selected_attempt_payload.get("attempt_id")
    final_debug_payload["final_status"] = "success"
    final_debug_payload["final_reason"] = None
    final_debug_payload["qc"] = selected_qc
    debug_dump_topology(run_dir, raw_image, final_debug_payload, suffix=None)

    result = infer_stage1_from_detections(
        chip_id=final_chip_id,
        raw_image=raw_image,
        gt_image=None,
        detections_raw=selected_geometry_detections,
        config=config,
        geometry_engine=effective_geometry_engine,
        quality_metrics=selected_qc,
        quality_gate_passed=bool(selected_qc.get("fit_success", False)),
        detection_mode="postprocess_topology_refined",
        retry_attempt=1 if selected_source != "json" else 0
    )

    output = save_stage1_result(
        result,
        run_dir,
        save_gt=False,
        save_individual_slices=save_individual_slices
    )

    if save_debug and result.debug_vis is not None:
        debug_path = run_dir / "debug_detection.png"
        cv2.imwrite(str(debug_path), result.debug_vis)
        logger.info(f"[{final_chip_id}] Debug visualization saved: debug_detection.png")

    logger.info(
        f"[{final_chip_id}] Stage1 postprocess-only output saved to: {run_dir} "
        f"(source={detections_json_path.name})"
    )
    return output


def run_stage1_postprocess_batch(
    input_dir: Path,
    output_dir: Path,
    config: Stage1Config,
    json_name: Optional[str] = None,
    min_topology_detections: Optional[int] = None,
    enable_fallback_detection: bool = True,
    save_individual_slices: bool = False,
    save_debug: bool = True
) -> List[Stage1Output]:
    """
    批量执行 Stage1 后处理（检测结果来自 JSON）。

    默认会递归查找：
    - yolo_raw_detections.json
    - adaptive_yolo_raw_detections.json
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if json_name:
        detection_jsons = sorted(input_dir.rglob(json_name))
    else:
        collected: List[Path] = []
        for name in ("yolo_raw_detections.json", "adaptive_yolo_raw_detections.json"):
            collected.extend(list(input_dir.rglob(name)))
        detection_jsons = sorted({str(p.resolve()): p for p in collected}.values(), key=str)

    if not detection_jsons:
        logger.warning(f"No detection json files found in {input_dir}")
        return []

    logger.info(f"Found {len(detection_jsons)} detection json files for postprocess-only batch")
    geometry_engine = CrossGeometryEngine(config.geometry)
    outputs: List[Stage1Output] = []

    for json_path in tqdm(detection_jsons, desc="Stage1 postprocess-only"):
        try:
            output = run_stage1_postprocess_from_json(
                detections_json_path=json_path,
                output_dir=output_dir,
                config=config,
                min_topology_detections=min_topology_detections,
                enable_fallback_detection=enable_fallback_detection,
                geometry_engine=geometry_engine,
                save_individual_slices=save_individual_slices,
                save_debug=save_debug
            )
            outputs.append(output)
        except Exception as e:
            logger.error(f"✗ postprocess failed for {json_path}: {e}")

    logger.info(
        "Stage1 postprocess-only batch complete: "
        f"{len(outputs)} success, {len(detection_jsons) - len(outputs)} failed"
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
