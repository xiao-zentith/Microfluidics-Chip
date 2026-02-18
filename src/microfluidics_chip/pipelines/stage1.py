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
import shutil
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from tqdm import tqdm
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback branch
    linear_sum_assignment = None

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
ARM_NAME_TO_INDICES = {
    "top": ARM_TEMPLATE_INDICES[0],
    "right": ARM_TEMPLATE_INDICES[1],
    "bottom": ARM_TEMPLATE_INDICES[2],
    "left": ARM_TEMPLATE_INDICES[3],
}
ALPHA_MATCH = 0.35
ALPHA_INLIER = 0.20
DET_KEEP_RATIO = 0.08
DET_REJECT_RATIO = 0.30
POST_QC_MAX_PURE_FILLED_RATIO = 0.67
POST_QC_MIN_MATCHED_COVERAGE_ARMS = 0.75
POST_QC_REQUIRE_BLANK_DETECTED = False
POST_QC_MIN_MATCHED_COUNT_BASE = 5
POST_QC_MIN_BLANK_CONFIDENCE = 0.01
POST_QC_MIN_BLANK_SCORE_SPREAD = 6.0
POST_QC_MAX_GEOMETRY_REPROJ_RATIO = 0.25
POST_QC_MAX_SLICE_CENTER_OFFSET_PX = 8.0
POST_QC_HARD_GEOMETRY_REPROJ_RATIO = 0.60
POST_QC_HARD_SLICE_CENTER_OFFSET_PX = 13.0
BLANK_MARGIN_THRESHOLD = 6.0
BLANK_FILLED_MARGIN_SCALE = 1.5
BLANK_FILLED_MIN_MATCHED_COUNT = 9
BLANK_FILLED_MIN_DET_USED_COUNT = 8

# Stage1-post 几何收敛阈值集中配置（避免散落补丁）
POSTPROCESS_RECOVERY_CFG: Dict[str, float] = {
    # Det cleanup
    "dedup_eps_ratio": 0.40,
    "dedup_eps_fallback_px": 10.0,
    "dedup_eps_min_px": 8.0,
    "dedup_eps_max_px": 48.0,
    "dedup_close_pair_ratio": 0.40,
    # Pitch
    "pitch_knn_k": 3.0,
    "pitch_small_dist_ratio": 0.50,
    "pitch_min_px": 20.0,
    "pitch_max_px": 120.0,
    "pitch_guard_low_px": 20.0,
    # Transform model selection
    "affine_force_gain": 0.80,
    "homography_force_gain": 0.75,
    "homography_projective_max_abs": 0.01,
    # Slice fallback trigger
    "fallback_reproj_ratio": 0.35,
    "fallback_min_real_points": 8.0,
    "fallback_max_fill_ratio": 0.33,
    "fallback_crop_radius_ratio": 0.55,
    "fallback_crop_radius_min_px": 12.0,
    "fallback_crop_radius_max_px": 96.0,
    # Low-coverage assignment recovery
    "relaxed_match_min_det": 12.0,
    "relaxed_match_trigger_count": 8.0,
    "relaxed_match_thresh_ratio": 1.20,
    "relaxed_reject_det_ratio": 1.20,
}


class TopologyPostprocessError(RuntimeError):
    """
    后处理阶段异常，携带可落盘的 debug payload。
    """
    def __init__(self, message: str, debug_payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.debug_payload = debug_payload or {}


def _classify_postprocess_failure_bucket(reason: Optional[str]) -> str:
    text = str(reason or "").lower()
    if "insufficient_detections_for_topology" in text:
        return "fail_n_det_lt_min"
    if "topology_fit_failed" in text:
        return "fail_topology_fit"
    if "geometry_alignment_failed" in text:
        return "fail_geometry_alignment"
    if "cannot read raw image" in text or "raw image" in text or "file not found" in text:
        return "fail_io_or_missing_raw"
    return "fail_io_or_missing_raw"


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


def _load_template_semantics(template_path: Optional[str]) -> Tuple[List[str], List[int], str]:
    """
    从自定义模板文件读取点ID与 blank 候选索引。
    返回 (template_ids, blank_candidate_indices, template_name)。
    """
    default_ids = [str(i) for i in range(12)]
    default_blank_candidates = [2, 5, 8, 11]
    default_name = "default_cross_template"

    if not template_path:
        return default_ids, default_blank_candidates, default_name

    try:
        p = Path(template_path)
        if not p.exists():
            return default_ids, default_blank_candidates, default_name
        payload = json.loads(p.read_text(encoding="utf-8"))
        geometry = payload.get("geometry", {}) if isinstance(payload, dict) else {}
        points = geometry.get("points", []) if isinstance(geometry, dict) else []
        ids: List[str] = []
        for pt in points:
            if isinstance(pt, dict) and "id" in pt:
                ids.append(str(pt["id"]))
        if len(ids) < 12:
            ids = default_ids
        else:
            ids = ids[:12]

        logic = payload.get("logic", {}) if isinstance(payload, dict) else {}
        blank_indices = logic.get("blank_candidate_indices")
        if not isinstance(blank_indices, list):
            blank_candidates = logic.get("blank_candidates", [])
            if isinstance(blank_candidates, list) and len(ids) >= 12:
                idx_map = {name: i for i, name in enumerate(ids)}
                blank_indices = [idx_map[c] for c in blank_candidates if c in idx_map]
            else:
                blank_indices = []
        blank_indices = [int(i) for i in blank_indices if isinstance(i, (int, float))][:4]
        if len(blank_indices) == 0:
            blank_indices = default_blank_candidates

        chip_meta = payload.get("chip_metadata", {}) if isinstance(payload, dict) else {}
        template_name = str(chip_meta.get("name") or p.name)
        return ids, blank_indices, template_name
    except Exception:
        return default_ids, default_blank_candidates, default_name


def _build_semantic_roles(
    template_ids: List[str],
    blank_idx: Optional[int]
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    基于 blank 所在臂，输出顺时针语义顺序和每个 template_id 的语义角色。
    """
    if len(template_ids) < 12:
        return [], {}, {}

    arms = [template_ids[i * 3:(i + 1) * 3] for i in range(4)]
    arm_start = max(0, min(3, int(blank_idx) // 3)) if blank_idx is not None else 0
    ordered_arm_indices = [(arm_start + k) % 4 for k in range(4)]

    semantic_order_clockwise_from_blank: List[str] = []
    for arm_idx in ordered_arm_indices:
        semantic_order_clockwise_from_blank.extend(arms[arm_idx])

    roles_by_template: Dict[str, str] = {}
    arm_role_by_template: Dict[str, str] = {}

    reference_roles = ["Enzyme-1", "Enzyme-2", "Liquid-only"]
    assay_names = ["Glucose", "Cholesterol", "UricAcid"]

    reference_arm = arms[ordered_arm_indices[0]]
    for i, tid in enumerate(reference_arm):
        role = reference_roles[min(i, len(reference_roles) - 1)]
        roles_by_template[tid] = role
        arm_role_by_template[tid] = "ReferenceArm"

    for arm_offset, assay in enumerate(assay_names, start=1):
        assay_arm = arms[ordered_arm_indices[arm_offset]]
        for rep_idx, tid in enumerate(assay_arm, start=1):
            roles_by_template[tid] = f"{assay}_rep{rep_idx}"
            arm_role_by_template[tid] = assay

    return semantic_order_clockwise_from_blank, roles_by_template, arm_role_by_template


def _arm_indices_from_template_ids(template_ids: List[str]) -> Dict[str, List[int]]:
    arms: Dict[str, List[int]] = {"Up": [], "Right": [], "Down": [], "Left": []}
    for idx, tid in enumerate(template_ids[:12]):
        t = str(tid).strip().upper()
        if t.startswith("U"):
            arms["Up"].append(idx)
        elif t.startswith("R"):
            arms["Right"].append(idx)
        elif t.startswith("D"):
            arms["Down"].append(idx)
        elif t.startswith("L"):
            arms["Left"].append(idx)
    if any(len(v) != 3 for v in arms.values()):
        return {
            "Up": list(ARM_TEMPLATE_INDICES[0]),
            "Right": list(ARM_TEMPLATE_INDICES[1]),
            "Down": list(ARM_TEMPLATE_INDICES[2]),
            "Left": list(ARM_TEMPLATE_INDICES[3]),
        }
    return arms


def _normalize_arm_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    text = str(name).strip().lower()
    mapping = {
        "up": "Up",
        "top": "Up",
        "right": "Right",
        "down": "Down",
        "bottom": "Down",
        "left": "Left",
    }
    return mapping.get(text)


def _select_blank_with_outermost_constraint(
    blank_indices: List[int],
    blank_scores: Dict[int, float],
    reference_arm: Optional[str],
    template_ids: List[str],
    candidate_indices: List[int],
) -> Tuple[Optional[int], List[int], bool]:
    candidates_outer = sorted(
        int(i) for i in candidate_indices
        if int(i) in set(OUTERMOST_TEMPLATE_INDICES) and 0 <= int(i) < 12
    )
    if not candidates_outer:
        candidates_outer = list(OUTERMOST_TEMPLATE_INDICES)
    arm_name = _normalize_arm_name(reference_arm)
    arms = _arm_indices_from_template_ids(template_ids)
    allowed = list(candidates_outer)
    if arm_name is not None and arm_name in arms and len(arms[arm_name]) > 0:
        arm_outer = int(arms[arm_name][-1])
        allowed = [arm_outer] if arm_outer in candidates_outer else [arm_outer]

    selected = None
    # 首先尝试沿用原 blank 预测（但必须属于 allowed）
    for idx in blank_indices:
        if int(idx) in allowed:
            selected = int(idx)
            break
    # 然后按 score 在 allowed 内挑最小
    if selected is None and blank_scores:
        scored_allowed = [(int(i), float(blank_scores[int(i)])) for i in allowed if int(i) in blank_scores]
        if scored_allowed:
            selected = int(min(scored_allowed, key=lambda kv: kv[1])[0])
    # 最后仍无结果时，不强行给值
    is_outer = bool(selected in OUTERMOST_TEMPLATE_INDICES) if selected is not None else False
    return selected, allowed, is_outer


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


def _compute_canonical_points_from_template(
    template_points: np.ndarray,
    canvas_size: int
) -> np.ndarray:
    pts = np.asarray(template_points, dtype=np.float32)
    centroid = np.mean(pts, axis=0, keepdims=True)
    centered = pts - centroid
    c = np.array([[canvas_size / 2.0, canvas_size / 2.0]], dtype=np.float32)
    return centered + c


def _estimate_affine_raw_to_canonical(
    raw_points: np.ndarray,
    canonical_points: np.ndarray,
    model_type: str = "similarity",
) -> Tuple[Optional[np.ndarray], str]:
    if raw_points.shape[0] < 3 or canonical_points.shape[0] < 3:
        return None, "none"

    model = str(model_type).strip().lower()
    if model in {"similarity", "sim"}:
        M, _ = cv2.estimateAffinePartial2D(
            raw_points.reshape(-1, 1, 2),
            canonical_points.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        if M is not None:
            return M.astype(np.float32), "similarity"
        return None, "none"
    if model == "affine":
        M, _ = cv2.estimateAffine2D(
            raw_points.reshape(-1, 1, 2),
            canonical_points.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        if M is not None:
            return M.astype(np.float32), "affine"
        return None, "none"
    if model == "homography" and raw_points.shape[0] >= 4:
        H, _ = cv2.findHomography(
            raw_points.reshape(-1, 1, 2),
            canonical_points.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        if H is not None:
            return H.astype(np.float32), "homography"
    return None, "none"


def _apply_transform_points(
    points: np.ndarray,
    matrix: np.ndarray,
    model_type: str,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    model = str(model_type).strip().lower()
    if model == "homography":
        if m.shape != (3, 3):
            return np.empty((0, 2), dtype=np.float32)
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), m).reshape(-1, 2)
        return out.astype(np.float32)
    if m.shape != (2, 3):
        return np.empty((0, 2), dtype=np.float32)
    out = cv2.transform(pts.reshape(-1, 1, 2), m).reshape(-1, 2)
    return out.astype(np.float32)


def _compute_transform_errors(
    raw_points: np.ndarray,
    canonical_points: np.ndarray,
    matrix: Optional[np.ndarray],
    model_type: str,
) -> Tuple[float, float, float, np.ndarray]:
    if matrix is None:
        return float("inf"), float("inf"), float("inf"), np.empty((0,), dtype=np.float32)
    pred = _apply_transform_points(raw_points, matrix, model_type)
    if pred.shape != canonical_points.shape:
        return float("inf"), float("inf"), float("inf"), np.empty((0,), dtype=np.float32)
    err = np.linalg.norm(pred - canonical_points.astype(np.float32), axis=1).astype(np.float32)
    if err.size == 0:
        return float("inf"), float("inf"), float("inf"), err
    return (
        float(np.median(err)),
        float(np.mean(err)),
        float(np.max(err)),
        err,
    )


def _fit_transform_model_with_score(
    raw_points: np.ndarray,
    canonical_points: np.ndarray,
    model_type: str,
) -> Dict[str, Any]:
    model = str(model_type).strip().lower()
    out: Dict[str, Any] = {
        "model": model,
        "matrix": None,
        "inliers_count": 0,
        "reproj_median": float("inf"),
        "reproj_mean": float("inf"),
        "reproj_max": float("inf"),
        "errors": np.empty((0,), dtype=np.float32),
        "valid": False,
    }
    if raw_points.shape[0] < 3:
        return out
    if model == "homography" and raw_points.shape[0] < 4:
        return out

    if model in {"similarity", "sim"}:
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            raw_points.reshape(-1, 1, 2),
            canonical_points.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        model = "similarity"
    elif model == "affine":
        matrix, inlier_mask = cv2.estimateAffine2D(
            raw_points.reshape(-1, 1, 2),
            canonical_points.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
    elif model == "homography":
        matrix, inlier_mask = cv2.findHomography(
            raw_points.reshape(-1, 1, 2),
            canonical_points.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
    else:
        return out

    out["model"] = model
    if matrix is None:
        return out
    inliers_count = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
    reproj_median, reproj_mean, reproj_max, errors = _compute_transform_errors(
        raw_points=raw_points,
        canonical_points=canonical_points,
        matrix=np.asarray(matrix, dtype=np.float32),
        model_type=model,
    )
    out.update(
        {
            "matrix": np.asarray(matrix, dtype=np.float32),
            "inliers_count": int(inliers_count),
            "reproj_median": float(reproj_median),
            "reproj_mean": float(reproj_mean),
            "reproj_max": float(reproj_max),
            "errors": errors,
            "valid": bool(np.isfinite(reproj_median)),
        }
    )
    return out


def _homography_sanity_check(matrix: Optional[np.ndarray], points: np.ndarray) -> bool:
    if matrix is None:
        return False
    H = np.asarray(matrix, dtype=np.float32)
    if H.shape != (3, 3):
        return False
    if not np.all(np.isfinite(H)):
        return False
    projective = float(max(abs(H[2, 0]), abs(H[2, 1])))
    if projective > float(POSTPROCESS_RECOVERY_CFG["homography_projective_max_abs"]):
        return False
    projected = _apply_transform_points(points, H, "homography")
    if projected.shape[0] != points.shape[0]:
        return False
    return bool(np.all(np.isfinite(projected)))


def _choose_transform_model(
    raw_points: np.ndarray,
    canonical_points: np.ndarray,
    point_sources: List[str],
) -> Dict[str, Any]:
    real_indices = [
        int(i)
        for i, s in enumerate(point_sources)
        if str(s) in {"det", "det_mid"}
    ]
    if len(real_indices) < 3:
        # 兜底：过少真实点时，退化为使用所有点拟合 similarity
        real_indices = list(range(min(raw_points.shape[0], canonical_points.shape[0])))

    raw_real = raw_points[np.asarray(real_indices, dtype=np.int32)].astype(np.float32)
    canonical_real = canonical_points[np.asarray(real_indices, dtype=np.int32)].astype(np.float32)

    sim = _fit_transform_model_with_score(raw_real, canonical_real, "similarity")
    aff = _fit_transform_model_with_score(raw_real, canonical_real, "affine")
    hom = _fit_transform_model_with_score(raw_real, canonical_real, "homography")
    if not _homography_sanity_check(hom.get("matrix"), raw_real):
        hom["valid"] = False
        hom["reproj_median"] = float("inf")
        hom["reproj_mean"] = float("inf")
        hom["reproj_max"] = float("inf")

    candidates = [c for c in (sim, aff, hom) if bool(c.get("valid"))]
    if not candidates:
        return {
            "model_chosen": "none",
            "matrix": None,
            "score_sim": float("inf"),
            "score_affine": float("inf"),
            "score_h": float("inf"),
            "inliers_count_sim": 0,
            "inliers_count_affine": 0,
            "inliers_count_h": 0,
            "reproj_median": float("inf"),
            "reproj_mean": float("inf"),
            "reproj_max": float("inf"),
            "real_indices": real_indices,
            "used_filled_points_for_transform": 0,
        }

    chosen = min(candidates, key=lambda c: float(c.get("reproj_median", float("inf"))))
    score_sim = float(sim.get("reproj_median", float("inf")))
    score_aff = float(aff.get("reproj_median", float("inf")))
    score_h = float(hom.get("reproj_median", float("inf")))

    # similarity 与 affine 同时可用时，允许强制 affine
    if np.isfinite(score_sim) and np.isfinite(score_aff):
        if score_aff < float(POSTPROCESS_RECOVERY_CFG["affine_force_gain"]) * score_sim:
            chosen = aff

    # homography 必须显著优于 affine/similarity 才启用
    base_best = min(
        score for score in [score_sim, score_aff] if np.isfinite(score)
    ) if any(np.isfinite(v) for v in [score_sim, score_aff]) else float("inf")
    if np.isfinite(score_h) and np.isfinite(base_best):
        if score_h < float(POSTPROCESS_RECOVERY_CFG["homography_force_gain"]) * base_best:
            chosen = hom

    return {
        "model_chosen": str(chosen.get("model", "none")),
        "matrix": chosen.get("matrix"),
        "score_sim": score_sim,
        "score_affine": score_aff,
        "score_h": score_h,
        "inliers_count_sim": int(sim.get("inliers_count", 0)),
        "inliers_count_affine": int(aff.get("inliers_count", 0)),
        "inliers_count_h": int(hom.get("inliers_count", 0)),
        "reproj_median": float(chosen.get("reproj_median", float("inf"))),
        "reproj_mean": float(chosen.get("reproj_mean", float("inf"))),
        "reproj_max": float(chosen.get("reproj_max", float("inf"))),
        "real_indices": real_indices,
        "used_filled_points_for_transform": 0,
    }


def _estimate_pitch_from_points(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 2:
        return float("nan")
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    nn = np.min(dmat, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return float("nan")
    return float(np.median(nn))


def _compute_knn_distances(points: np.ndarray, k: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return np.empty((0,), dtype=np.float32)
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    k_eff = int(max(1, min(int(k), pts.shape[0] - 1)))
    k_small = np.partition(dmat, kth=k_eff - 1, axis=1)[:, :k_eff]
    flat = k_small.reshape(-1)
    flat = flat[np.isfinite(flat) & (flat > 0.0)]
    return flat.astype(np.float32)


def _estimate_pitch_robust_from_points(points: np.ndarray) -> Dict[str, Any]:
    pts = np.asarray(points, dtype=np.float32)
    out = {
        "pitch_raw": float("nan"),
        "pitch_filtered": float("nan"),
        "discard_threshold": float("nan"),
        "k_neighbors": int(POSTPROCESS_RECOVERY_CFG["pitch_knn_k"]),
        "n_samples": int(0),
    }
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return out
    d = _compute_knn_distances(pts, int(POSTPROCESS_RECOVERY_CFG["pitch_knn_k"]))
    out["n_samples"] = int(d.size)
    if d.size == 0:
        return out
    med = float(np.median(d))
    p10 = float(np.percentile(d, 10))
    small_cut = float(POSTPROCESS_RECOVERY_CFG["pitch_small_dist_ratio"] * med)
    discard_thr = float(max(p10, small_cut))
    d_filtered = d[d >= discard_thr]
    if d_filtered.size == 0:
        d_filtered = d
    out["pitch_raw"] = float(med)
    out["pitch_filtered"] = float(np.median(d_filtered))
    out["discard_threshold"] = float(discard_thr)
    return out


def _estimate_pitch_from_roi_or_fallback(
    roi_bbox: Optional[Any],
    geometry_engine: CrossGeometryEngine,
) -> Tuple[float, str]:
    if isinstance(roi_bbox, (list, tuple)) and len(roi_bbox) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in roi_bbox]
            rw = max(1.0, x2 - x1)
            rh = max(1.0, y2 - y1)
            pitch = float(min(rw, rh) / 6.0)
            if np.isfinite(pitch) and pitch > 1.0:
                return pitch, "roi_extent_over_6"
        except Exception:
            pass
    fallback = float(max(8.0, geometry_engine.config.ideal_chamber_step))
    return fallback, "geometry_step_fallback"


def _build_detection_clusters(
    source_detections: List[ChamberDetection],
    eps_px: float,
) -> Tuple[List[ChamberDetection], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not source_detections:
        return [], [], []
    centers = np.asarray([d.center for d in source_detections], dtype=np.float32)
    n = int(centers.shape[0])
    visited = np.zeros((n,), dtype=bool)
    clusters: List[List[int]] = []

    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    for seed in range(n):
        if visited[seed]:
            continue
        q = [seed]
        visited[seed] = True
        comp: List[int] = []
        while q:
            idx = q.pop()
            comp.append(int(idx))
            neighbors = np.where(dmat[idx] <= float(eps_px))[0].tolist()
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    q.append(int(nb))
        comp.sort()
        clusters.append(comp)

    merged: List[ChamberDetection] = []
    merged_meta: List[Dict[str, Any]] = []
    cluster_rows: List[Dict[str, Any]] = []
    for cid, members in enumerate(clusters):
        pts = np.asarray([source_detections[i].center for i in members], dtype=np.float32)
        confs = np.asarray(
            [max(float(source_detections[i].confidence), 1e-6) for i in members],
            dtype=np.float32,
        )
        w = confs / np.sum(confs)
        center = np.sum(pts * w[:, None], axis=0).astype(np.float32)
        rep_idx = max(members, key=lambda i: float(source_detections[i].confidence))
        rep = source_detections[rep_idx]
        merged_conf = float(np.max(confs))
        merged_det = ChamberDetection(
            bbox=rep.bbox,
            center=(float(center[0]), float(center[1])),
            class_id=int(rep.class_id),
            confidence=float(merged_conf),
        )
        merged.append(merged_det)
        merged_meta.append(
            {
                "cluster_id": int(cid),
                "members_idx": [int(i) for i in members],
                "representative_idx": int(rep_idx),
                "cluster_size": int(len(members)),
            }
        )
        cluster_rows.append(
            {
                "cluster_id": int(cid),
                "members_idx": [int(i) for i in members],
                "cluster_size": int(len(members)),
                "merged_center": [float(center[0]), float(center[1])],
                "merged_conf": float(merged_conf),
                "representative_idx": int(rep_idx),
            }
        )
    return merged, merged_meta, cluster_rows


def _prepare_postprocess_detections(
    source_detections: List[ChamberDetection],
    roi_bbox: Optional[Any],
    geometry_engine: CrossGeometryEngine,
) -> Dict[str, Any]:
    n_det_raw = int(len(source_detections))
    raw_centers = (
        np.asarray([d.center for d in source_detections], dtype=np.float32)
        if source_detections
        else np.empty((0, 2), dtype=np.float32)
    )
    pitch_raw_info = _estimate_pitch_robust_from_points(raw_centers)
    pitch_raw = float(pitch_raw_info.get("pitch_filtered", float("nan")))
    if not np.isfinite(pitch_raw):
        pitch_raw = float(pitch_raw_info.get("pitch_raw", float("nan")))
    pitch_fb, pitch_fb_method = _estimate_pitch_from_roi_or_fallback(roi_bbox=roi_bbox, geometry_engine=geometry_engine)
    pitch_init = float(pitch_raw if np.isfinite(pitch_raw) else pitch_fb)

    dedup_eps = float(
        np.clip(
            float(POSTPROCESS_RECOVERY_CFG["dedup_eps_ratio"]) * pitch_init,
            float(POSTPROCESS_RECOVERY_CFG["dedup_eps_min_px"]),
            float(POSTPROCESS_RECOVERY_CFG["dedup_eps_max_px"]),
        )
    ) if np.isfinite(pitch_init) else float(POSTPROCESS_RECOVERY_CFG["dedup_eps_fallback_px"])

    has_close_pair = False
    min_pair_dist = float("inf")
    if raw_centers.shape[0] >= 2:
        dmat = np.linalg.norm(raw_centers[:, None, :] - raw_centers[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        min_pair_dist = float(np.min(dmat))
        close_thr = float(
            max(
                POSTPROCESS_RECOVERY_CFG["dedup_eps_fallback_px"],
                POSTPROCESS_RECOVERY_CFG["dedup_close_pair_ratio"] * (pitch_init if np.isfinite(pitch_init) else 25.0),
            )
        )
        has_close_pair = bool(np.isfinite(min_pair_dist) and min_pair_dist <= close_thr)

    pitch_guard_triggered = False
    if n_det_raw > 12 and (not np.isfinite(pitch_raw) or pitch_raw < float(POSTPROCESS_RECOVERY_CFG["pitch_guard_low_px"])):
        pitch_guard_triggered = True

    trigger_cleanup = bool(n_det_raw > 12 or has_close_pair)
    clusters: List[Dict[str, Any]] = []
    dedup_meta: List[Dict[str, Any]] = []
    if trigger_cleanup and n_det_raw > 0:
        cleaned_dets, dedup_meta, clusters = _build_detection_clusters(
            source_detections=source_detections,
            eps_px=dedup_eps,
        )
    else:
        cleaned_dets = list(source_detections)
        for idx in range(n_det_raw):
            det = source_detections[idx]
            dedup_meta.append(
                {
                    "cluster_id": int(idx),
                    "members_idx": [int(idx)],
                    "representative_idx": int(idx),
                    "cluster_size": 1,
                }
            )
            clusters.append(
                {
                    "cluster_id": int(idx),
                    "members_idx": [int(idx)],
                    "cluster_size": 1,
                    "merged_center": [float(det.center[0]), float(det.center[1])],
                    "merged_conf": float(det.confidence),
                    "representative_idx": int(idx),
                }
            )

    dedup_centers = (
        np.asarray([d.center for d in cleaned_dets], dtype=np.float32)
        if cleaned_dets
        else np.empty((0, 2), dtype=np.float32)
    )
    pitch_filtered_info = _estimate_pitch_robust_from_points(dedup_centers)
    pitch_filtered = float(pitch_filtered_info.get("pitch_filtered", float("nan")))
    if not np.isfinite(pitch_filtered):
        pitch_filtered = float(pitch_filtered_info.get("pitch_raw", float("nan")))
    if not np.isfinite(pitch_filtered):
        pitch_filtered = float(pitch_raw if np.isfinite(pitch_raw) else pitch_fb)
    pitch_final = float(np.clip(
        pitch_filtered,
        float(POSTPROCESS_RECOVERY_CFG["pitch_min_px"]),
        float(POSTPROCESS_RECOVERY_CFG["pitch_max_px"]),
    ))
    if not np.isfinite(pitch_filtered) or abs(pitch_final - pitch_filtered) > 1e-6:
        pitch_guard_triggered = True
    if n_det_raw > 12 and pitch_final < float(POSTPROCESS_RECOVERY_CFG["pitch_guard_low_px"]):
        pitch_guard_triggered = True

    pitch_method = "knn3_filtered_dedup" if trigger_cleanup else "knn3_filtered"
    if (not np.isfinite(pitch_filtered)) and np.isfinite(pitch_fb):
        pitch_method = f"{pitch_method}|{pitch_fb_method}"

    return {
        "detections_clean": cleaned_dets,
        "dedup_meta": dedup_meta,
        "clusters": clusters,
        "n_det_raw": n_det_raw,
        "n_det_dedup": int(len(cleaned_dets)),
        "trigger_cleanup": bool(trigger_cleanup),
        "has_close_pair": bool(has_close_pair),
        "min_pair_dist_px": None if not np.isfinite(min_pair_dist) else float(min_pair_dist),
        "dedup_eps_px": float(dedup_eps),
        "pitch_raw": float(pitch_raw_info.get("pitch_raw", float("nan"))),
        "pitch_filtered": float(pitch_filtered),
        "pitch_final": float(pitch_final),
        "pitch_guard_triggered": bool(pitch_guard_triggered),
        "pitch_method": str(pitch_method),
        "pitch_raw_debug": pitch_raw_info,
        "pitch_filtered_debug": pitch_filtered_info,
    }


def _build_final_points_det_priority(
    fitted_centers: np.ndarray,
    source_detections: List[ChamberDetection],
    match_map: Dict[int, int],
    pitch_px: float,
    keep_det_ratio: float = DET_KEEP_RATIO,
    reject_det_ratio: float = DET_REJECT_RATIO,
) -> Tuple[np.ndarray, List[str], List[Optional[float]], Dict[str, Any]]:
    n = min(12, int(fitted_centers.shape[0]))
    fitted = np.asarray(fitted_centers[:n], dtype=np.float32)
    det_centers = (
        np.array([d.center for d in source_detections], dtype=np.float32)
        if source_detections
        else np.empty((0, 2), dtype=np.float32)
    )

    keep_thr = float(max(2.0, keep_det_ratio * pitch_px))
    reject_thr = float(max(4.0, reject_det_ratio * pitch_px))

    final_points = fitted.copy()
    point_sources: List[str] = []
    det_fit_distance_px: List[Optional[float]] = []
    matched_det_indices: List[int] = []
    det_points: List[Optional[List[float]]] = []
    fit_points: List[List[float]] = []
    final_points_list: List[List[float]] = []

    det_used_count = 0
    det_mid_count = 0
    reject_det_count = 0
    fill_count = 0
    used_dists: List[float] = []

    for idx in range(n):
        p_fit = fitted[idx]
        fit_points.append([float(p_fit[0]), float(p_fit[1])])
        det_idx = int(match_map[idx]) if idx in match_map else -1

        if det_idx >= 0 and det_idx < int(det_centers.shape[0]):
            p_det = det_centers[det_idx]
            d = float(np.linalg.norm(p_det - p_fit))
            det_fit_distance_px.append(d)
            matched_det_indices.append(det_idx)
            det_points.append([float(p_det[0]), float(p_det[1])])
            used_dists.append(d)

            if d <= keep_thr:
                final_points[idx] = p_det
                point_sources.append("det")
                det_used_count += 1
            elif d >= reject_thr:
                final_points[idx] = p_fit
                point_sources.append("filled")
                reject_det_count += 1
            else:
                final_points[idx] = p_det
                point_sources.append("det_mid")
                det_used_count += 1
                det_mid_count += 1
        else:
            det_fit_distance_px.append(None)
            matched_det_indices.append(-1)
            det_points.append(None)
            final_points[idx] = p_fit
            point_sources.append("filled")
            fill_count += 1

        final_points_list.append([float(final_points[idx, 0]), float(final_points[idx, 1])])

    det_fit_dist_mean_px = float(np.mean(used_dists)) if used_dists else None
    det_fit_dist_max_px = float(np.max(used_dists)) if used_dists else None

    stats = {
        "keep_det_thr_px": keep_thr,
        "reject_det_thr_px": reject_thr,
        "det_used_count": int(det_used_count),
        "det_mid_count": int(det_mid_count),
        "reject_det_count": int(reject_det_count),
        "fill_count": int(fill_count),
        "det_fit_dist_mean_px": det_fit_dist_mean_px,
        "det_fit_dist_max_px": det_fit_dist_max_px,
        "matched_det_indices": matched_det_indices,
        "det_points": det_points,
        "fit_points": fit_points,
        "final_points": final_points_list,
    }
    return final_points, point_sources, det_fit_distance_px, stats


def _compute_blank_brightness_outermost(
    canonical_image: np.ndarray,
    canonical_centers: np.ndarray,
    candidate_indices: List[int],
    chamber_radius_px: float,
) -> Tuple[List[int], Dict[int, float], float, float, Dict[int, Dict[str, float]]]:
    """
    旧逻辑对照：仅在 outermost 候选中选择亮度最小者。
    """
    h, w = canonical_image.shape[:2]
    hsv = cv2.cvtColor(canonical_image, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)

    r = float(max(4.0, chamber_radius_px))
    r_in = 0.55 * r
    blank_scores: Dict[int, float] = {}
    blank_features: Dict[int, Dict[str, float]] = {}

    for idx in candidate_indices:
        if idx < 0 or idx >= canonical_centers.shape[0]:
            continue
        cx, cy = float(canonical_centers[idx, 0]), float(canonical_centers[idx, 1])
        x1 = int(max(0, math.floor(cx - r_in - 2)))
        y1 = int(max(0, math.floor(cy - r_in - 2)))
        x2 = int(min(w, math.ceil(cx + r_in + 2)))
        y2 = int(min(h, math.ceil(cy + r_in + 2)))
        if x2 <= x1 or y2 <= y1:
            continue

        local_V = V[y1:y2, x1:x2]
        if local_V.size == 0:
            continue

        yy, xx = np.ogrid[: local_V.shape[0], : local_V.shape[1]]
        local_cx = cx - x1
        local_cy = cy - y1
        dist2 = (xx - local_cx) ** 2 + (yy - local_cy) ** 2
        inner_mask = dist2 <= (r_in ** 2)
        if int(np.sum(inner_mask)) < 20:
            continue

        pixels = local_V[inner_mask]
        p10 = float(np.percentile(pixels, 10))
        med = float(np.median(pixels))
        score = float(0.7 * p10 + 0.3 * med)
        blank_scores[int(idx)] = score
        blank_features[int(idx)] = {
            "V_low": p10,
            "V_med": med,
            "score": score,
            "inner_radius_px": float(r_in),
            "center_canonical_x": float(cx),
            "center_canonical_y": float(cy),
        }

    if not blank_scores:
        return [], {}, 0.0, 0.0, blank_features

    blank_idx = int(min(blank_scores, key=blank_scores.get))
    sorted_scores = sorted(float(v) for v in blank_scores.values())
    score_spread = float(sorted_scores[1] - sorted_scores[0]) if len(sorted_scores) > 1 else 0.0
    blank_confidence = (
        float(np.clip(score_spread / (sorted_scores[1] + 1e-6), 0.0, 1.0))
        if len(sorted_scores) > 1
        else 0.0
    )
    return [blank_idx], blank_scores, blank_confidence, score_spread, blank_features


def _compute_blank_color_mode(
    canonical_image: np.ndarray,
    canonical_centers: np.ndarray,
    template_ids: List[str],
    w1: float,
    w2: float,
    w3: float,
    arm_margin_thr: float,
    blank_margin_thr: float,
    chamber_radius_px: float,
) -> Dict[str, Any]:
    """
    新逻辑：基于 R/G + 红度 + 饱和度差分判定 reference_arm 与 BLANK。
    """
    h, w = canonical_image.shape[:2]
    rgb = cv2.cvtColor(canonical_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(canonical_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    S = hsv[:, :, 1] / 255.0
    eps = 1e-6

    r = float(max(4.0, chamber_radius_px))
    r_in1, r_in2 = 0.25 * r, 0.55 * r
    r_out1, r_out2 = 1.10 * r, 1.40 * r

    chamber_scores_idx: Dict[int, float] = {}
    chamber_features_by_idx: Dict[int, Dict[str, float]] = {}

    for idx in range(min(12, canonical_centers.shape[0])):
        cx, cy = float(canonical_centers[idx, 0]), float(canonical_centers[idx, 1])
        x1 = int(max(0, math.floor(cx - r_out2 - 2)))
        y1 = int(max(0, math.floor(cy - r_out2 - 2)))
        x2 = int(min(w, math.ceil(cx + r_out2 + 2)))
        y2 = int(min(h, math.ceil(cy + r_out2 + 2)))
        if x2 <= x1 or y2 <= y1:
            continue

        local_R = R[y1:y2, x1:x2]
        local_G = G[y1:y2, x1:x2]
        local_B = B[y1:y2, x1:x2]
        local_S = S[y1:y2, x1:x2]
        if local_R.size == 0:
            continue

        yy, xx = np.ogrid[: local_R.shape[0], : local_R.shape[1]]
        local_cx = cx - x1
        local_cy = cy - y1
        dist2 = (xx - local_cx) ** 2 + (yy - local_cy) ** 2
        inner_mask = (dist2 >= (r_in1 ** 2)) & (dist2 <= (r_in2 ** 2))
        ring_mask = (dist2 >= (r_out1 ** 2)) & (dist2 <= (r_out2 ** 2))
        if int(np.sum(inner_mask)) < 20 or int(np.sum(ring_mask)) < 20:
            continue

        R_in = float(np.median(local_R[inner_mask]))
        G_in = float(np.median(local_G[inner_mask]))
        B_in = float(np.median(local_B[inner_mask]))
        S_in = float(np.median(local_S[inner_mask]))

        R_out = float(np.median(local_R[ring_mask]))
        G_out = float(np.median(local_G[ring_mask]))
        B_out = float(np.median(local_B[ring_mask]))
        S_out = float(np.median(local_S[ring_mask]))

        f_rg_in = float(np.log((R_in + eps) / (G_in + eps)))
        f_rg_out = float(np.log((R_out + eps) / (G_out + eps)))
        f_red_in = float((R_in - B_in) / (R_in + G_in + B_in + eps))
        f_red_out = float((R_out - B_out) / (R_out + G_out + B_out + eps))
        f_s_in = float(S_in)
        f_s_out = float(S_out)

        d_f_rg = float(f_rg_in - f_rg_out)
        d_f_red = float(f_red_in - f_red_out)
        d_f_s = float(f_s_in - f_s_out)
        score_ch = float(w1 * abs(d_f_rg) + w2 * d_f_red + w3 * d_f_s)

        chamber_scores_idx[idx] = score_ch
        chamber_features_by_idx[idx] = {
            "delta_f_rg": d_f_rg,
            "delta_f_red": d_f_red,
            "delta_f_s": d_f_s,
            "score_ch": score_ch,
            "R_inner_med": R_in,
            "G_inner_med": G_in,
            "B_inner_med": B_in,
            "R_ring_med": R_out,
            "G_ring_med": G_out,
            "B_ring_med": B_out,
            "S_inner_med": S_in,
            "S_ring_med": S_out,
            "inner_radius_min_px": float(r_in1),
            "inner_radius_max_px": float(r_in2),
            "ring_radius_min_px": float(r_out1),
            "ring_radius_max_px": float(r_out2),
            "center_canonical_x": float(cx),
            "center_canonical_y": float(cy),
        }

    # arm groups by template id prefix, fallback to default index layout
    arms_by_name: Dict[str, List[int]] = {"Up": [], "Right": [], "Down": [], "Left": []}
    for idx, tid in enumerate(template_ids[:12]):
        t = str(tid).strip().upper()
        if t.startswith("U"):
            arms_by_name["Up"].append(idx)
        elif t.startswith("R"):
            arms_by_name["Right"].append(idx)
        elif t.startswith("D"):
            arms_by_name["Down"].append(idx)
        elif t.startswith("L"):
            arms_by_name["Left"].append(idx)

    if any(len(v) != 3 for v in arms_by_name.values()):
        arms_by_name = {
            "Up": list(ARM_TEMPLATE_INDICES[0]),
            "Right": list(ARM_TEMPLATE_INDICES[1]),
            "Down": list(ARM_TEMPLATE_INDICES[2]),
            "Left": list(ARM_TEMPLATE_INDICES[3]),
        }

    arm_scores: Dict[str, float] = {}
    arm_chamber_scores: Dict[str, Dict[str, float]] = {}
    for arm_name, idxs in arms_by_name.items():
        vals: List[float] = []
        chamber_dict: Dict[str, float] = {}
        for idx in idxs:
            if idx not in chamber_scores_idx:
                continue
            score = float(chamber_scores_idx[idx])
            vals.append(score)
            chamber_dict[str(template_ids[idx])] = score
        arm_chamber_scores[arm_name] = chamber_dict
        arm_scores[arm_name] = float(np.median(vals)) if vals else float("inf")

    finite_arms = {k: v for k, v in arm_scores.items() if np.isfinite(v)}
    if not finite_arms:
        return {
            "blank_idx_color": None,
            "blank_status_color": "unresolved",
            "blank_confidence_color": 0.0,
            "blank_margin": 0.0,
            "arm_margin": 0.0,
            "reference_arm": None,
            "arm_scores": arm_scores,
            "reference_arm_chamber_scores": {},
            "chamber_scores_by_index": chamber_scores_idx,
            "chamber_features_by_index": chamber_features_by_idx,
            "arm_margin_thr": float(arm_margin_thr),
            "blank_margin_thr": float(blank_margin_thr),
            "mode": "color",
            "reason": "arm_scores_unavailable",
        }

    sorted_arms = sorted(finite_arms.items(), key=lambda kv: kv[1])
    reference_arm = str(sorted_arms[0][0])
    arm_margin = (
        float(sorted_arms[1][1] - sorted_arms[0][1])
        if len(sorted_arms) > 1
        else 0.0
    )

    ref_indices = arms_by_name.get(reference_arm, [])
    ref_scores: List[Tuple[int, float]] = [
        (idx, float(chamber_scores_idx[idx]))
        for idx in ref_indices
        if idx in chamber_scores_idx
    ]
    if not ref_scores:
        return {
            "blank_idx_color": None,
            "blank_status_color": "unresolved",
            "blank_confidence_color": 0.0,
            "blank_margin": 0.0,
            "arm_margin": arm_margin,
            "reference_arm": reference_arm,
            "arm_scores": arm_scores,
            "reference_arm_chamber_scores": arm_chamber_scores.get(reference_arm, {}),
            "chamber_scores_by_index": chamber_scores_idx,
            "chamber_features_by_index": chamber_features_by_idx,
            "arm_margin_thr": float(arm_margin_thr),
            "blank_margin_thr": float(blank_margin_thr),
            "mode": "color",
            "reason": "reference_arm_chambers_unavailable",
        }

    ref_scores_sorted = sorted(ref_scores, key=lambda x: x[1])
    blank_idx_color = int(ref_scores_sorted[0][0])
    blank_margin = (
        float(ref_scores_sorted[1][1] - ref_scores_sorted[0][1])
        if len(ref_scores_sorted) > 1
        else 0.0
    )
    blank_confidence_color = (
        float(np.clip(blank_margin / (abs(ref_scores_sorted[1][1]) + 1e-6), 0.0, 1.0))
        if len(ref_scores_sorted) > 1
        else 0.0
    )

    blank_status_color = "confirmed"
    reasons: List[str] = []
    if arm_margin < float(arm_margin_thr):
        blank_status_color = "unresolved"
        reasons.append(f"arm_margin<{arm_margin_thr:.3f}")
    if blank_margin < float(blank_margin_thr):
        blank_status_color = "unresolved"
        reasons.append(f"blank_margin<{blank_margin_thr:.3f}")

    return {
        "blank_idx_color": blank_idx_color,
        "blank_status_color": blank_status_color,
        "blank_confidence_color": blank_confidence_color,
        "blank_margin": blank_margin,
        "arm_margin": arm_margin,
        "reference_arm": reference_arm,
        "arm_scores": arm_scores,
        "reference_arm_chamber_scores": arm_chamber_scores.get(reference_arm, {}),
        "chamber_scores_by_index": chamber_scores_idx,
        "chamber_features_by_index": chamber_features_by_idx,
        "arm_margin_thr": float(arm_margin_thr),
        "blank_margin_thr": float(blank_margin_thr),
        "mode": "color",
        "reasons": reasons,
    }


def _compute_blank_chromaticity_mode(
    canonical_image: np.ndarray,
    canonical_centers: np.ndarray,
    template_ids: List[str],
    w1: float,
    w2: float,
    w3: float,
    w4: float,
    arm_margin_thr: float,
    blank_margin_thr: float,
    clip_quantile: float,
    clip_rg_min: float,
    clip_re_min: float,
    chamber_radius_px: float,
) -> Dict[str, Any]:
    """
    v2 新逻辑：基于 chromaticity/ratio/red-excess/saturation 的 reaction-ness 判定。
    目标是先选 reference arm（整体反应最弱），再在该臂选 blank。
    """
    h, w = canonical_image.shape[:2]
    rgb = cv2.cvtColor(canonical_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(canonical_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    S = hsv[:, :, 1] / 255.0
    eps = 1e-6

    r = float(max(4.0, chamber_radius_px))
    r_in1, r_in2 = 0.25 * r, 0.55 * r
    r_out1, r_out2 = 1.10 * r, 1.40 * r

    # 先收集原始差分特征，再做鲁棒裁剪
    chamber_raw: Dict[int, Dict[str, float]] = {}
    for idx in range(min(12, canonical_centers.shape[0])):
        cx, cy = float(canonical_centers[idx, 0]), float(canonical_centers[idx, 1])
        x1 = int(max(0, math.floor(cx - r_out2 - 2)))
        y1 = int(max(0, math.floor(cy - r_out2 - 2)))
        x2 = int(min(w, math.ceil(cx + r_out2 + 2)))
        y2 = int(min(h, math.ceil(cy + r_out2 + 2)))
        if x2 <= x1 or y2 <= y1:
            continue

        local_R = R[y1:y2, x1:x2]
        local_G = G[y1:y2, x1:x2]
        local_B = B[y1:y2, x1:x2]
        local_S = S[y1:y2, x1:x2]
        if local_R.size == 0:
            continue

        yy, xx = np.ogrid[: local_R.shape[0], : local_R.shape[1]]
        local_cx = cx - x1
        local_cy = cy - y1
        dist2 = (xx - local_cx) ** 2 + (yy - local_cy) ** 2
        donut_mask = (dist2 >= (r_in1 ** 2)) & (dist2 <= (r_in2 ** 2))
        ring_mask = (dist2 >= (r_out1 ** 2)) & (dist2 <= (r_out2 ** 2))
        if int(np.sum(donut_mask)) < 20 or int(np.sum(ring_mask)) < 20:
            continue

        R_d = float(np.median(local_R[donut_mask]))
        G_d = float(np.median(local_G[donut_mask]))
        B_d = float(np.median(local_B[donut_mask]))
        S_d = float(np.median(local_S[donut_mask]))

        R_r = float(np.median(local_R[ring_mask]))
        G_r = float(np.median(local_G[ring_mask]))
        B_r = float(np.median(local_B[ring_mask]))
        S_r = float(np.median(local_S[ring_mask]))

        sum_d = R_d + G_d + B_d
        sum_r = R_r + G_r + B_r
        chroma_r_d = float(R_d / (sum_d + eps))
        chroma_r_r = float(R_r / (sum_r + eps))
        rg_d = float(R_d / (G_d + eps))
        rg_r = float(R_r / (G_r + eps))
        re_d = float(R_d - 0.5 * (G_d + B_d))
        re_r = float(R_r - 0.5 * (G_r + B_r))

        chamber_raw[idx] = {
            "R_donut_med": R_d,
            "G_donut_med": G_d,
            "B_donut_med": B_d,
            "S_donut_med": S_d,
            "R_ring_med": R_r,
            "G_ring_med": G_r,
            "B_ring_med": B_r,
            "S_ring_med": S_r,
            "df_r_raw": float(chroma_r_d - chroma_r_r),
            "df_rg_raw": float(rg_d - rg_r),
            "df_re_raw": float(re_d - re_r),
            "df_s_raw": float(S_d - S_r),
            "center_canonical_x": float(cx),
            "center_canonical_y": float(cy),
            "donut_radius_min_px": float(r_in1),
            "donut_radius_max_px": float(r_in2),
            "ring_radius_min_px": float(r_out1),
            "ring_radius_max_px": float(r_out2),
        }

    if not chamber_raw:
        return {
            "blank_idx_chromaticity": None,
            "blank_status_chromaticity": "unresolved",
            "blank_confidence_chromaticity": 0.0,
            "blank_margin": 0.0,
            "arm_margin": 0.0,
            "reference_arm": None,
            "arm_scores": {},
            "reference_arm_chamber_scores": {},
            "chamber_scores_by_index": {},
            "chamber_features_by_index": {},
            "arm_margin_thr": float(arm_margin_thr),
            "blank_margin_thr": float(blank_margin_thr),
            "clip_quantile": float(clip_quantile),
            "clip_rg": float(clip_rg_min),
            "clip_re": float(clip_re_min),
            "arm_top2": [],
            "blank_top2": [],
            "mode": "chromaticity",
            "reasons": ["chamber_features_unavailable"],
            "score_method": "w1*abs(df_r) + w2*abs(df_rg) + w3*max(df_re,0) + w4*max(df_s,0)",
            "weights": {"w1": float(w1), "w2": float(w2), "w3": float(w3), "w4": float(w4)},
        }

    # 鲁棒裁剪阈值：max(min_clip, p99(|df|))
    rg_abs = np.asarray([abs(v["df_rg_raw"]) for v in chamber_raw.values()], dtype=np.float32)
    re_abs = np.asarray([abs(v["df_re_raw"]) for v in chamber_raw.values()], dtype=np.float32)
    q = float(np.clip(clip_quantile, 0.5, 1.0))
    clip_rg = float(max(clip_rg_min, np.quantile(rg_abs, q) if rg_abs.size > 0 else clip_rg_min))
    clip_re = float(max(clip_re_min, np.quantile(re_abs, q) if re_abs.size > 0 else clip_re_min))

    chamber_scores_idx: Dict[int, float] = {}
    chamber_features_by_idx: Dict[int, Dict[str, float]] = {}
    for idx, feat in chamber_raw.items():
        df_r = float(feat["df_r_raw"])
        df_rg = float(np.clip(float(feat["df_rg_raw"]), -clip_rg, clip_rg))
        df_re = float(np.clip(float(feat["df_re_raw"]), -clip_re, clip_re))
        df_s = float(feat["df_s_raw"])
        reaction_score = float(
            w1 * abs(df_r) +
            w2 * abs(df_rg) +
            w3 * max(df_re, 0.0) +
            w4 * max(df_s, 0.0)
        )
        chamber_scores_idx[int(idx)] = reaction_score
        chamber_features_by_idx[int(idx)] = {
            **{k: float(v) for k, v in feat.items()},
            "df_r": df_r,
            "df_rg": df_rg,
            "df_re": df_re,
            "df_s": df_s,
            "clip_rg": clip_rg,
            "clip_re": clip_re,
            "reaction_score": reaction_score,
        }

    arms_by_name: Dict[str, List[int]] = {"Up": [], "Right": [], "Down": [], "Left": []}
    for idx, tid in enumerate(template_ids[:12]):
        t = str(tid).strip().upper()
        if t.startswith("U"):
            arms_by_name["Up"].append(idx)
        elif t.startswith("R"):
            arms_by_name["Right"].append(idx)
        elif t.startswith("D"):
            arms_by_name["Down"].append(idx)
        elif t.startswith("L"):
            arms_by_name["Left"].append(idx)
    if any(len(v) != 3 for v in arms_by_name.values()):
        arms_by_name = {
            "Up": list(ARM_TEMPLATE_INDICES[0]),
            "Right": list(ARM_TEMPLATE_INDICES[1]),
            "Down": list(ARM_TEMPLATE_INDICES[2]),
            "Left": list(ARM_TEMPLATE_INDICES[3]),
        }

    arm_scores: Dict[str, float] = {}
    arm_chamber_scores: Dict[str, Dict[str, float]] = {}
    for arm_name, idxs in arms_by_name.items():
        vals: List[float] = []
        chamber_dict: Dict[str, float] = {}
        for idx in idxs:
            if idx not in chamber_scores_idx:
                continue
            score = float(chamber_scores_idx[idx])
            vals.append(score)
            chamber_dict[str(template_ids[idx])] = score
        arm_chamber_scores[arm_name] = chamber_dict
        arm_scores[arm_name] = float(np.median(vals)) if vals else float("inf")

    finite_arms = {k: v for k, v in arm_scores.items() if np.isfinite(v)}
    if not finite_arms:
        return {
            "blank_idx_chromaticity": None,
            "blank_status_chromaticity": "unresolved",
            "blank_confidence_chromaticity": 0.0,
            "blank_margin": 0.0,
            "arm_margin": 0.0,
            "reference_arm": None,
            "arm_scores": arm_scores,
            "reference_arm_chamber_scores": {},
            "chamber_scores_by_index": chamber_scores_idx,
            "chamber_features_by_index": chamber_features_by_idx,
            "arm_margin_thr": float(arm_margin_thr),
            "blank_margin_thr": float(blank_margin_thr),
            "clip_quantile": q,
            "clip_rg": clip_rg,
            "clip_re": clip_re,
            "arm_top2": [],
            "blank_top2": [],
            "mode": "chromaticity",
            "reasons": ["arm_scores_unavailable"],
            "score_method": "w1*abs(df_r) + w2*abs(df_rg) + w3*max(df_re,0) + w4*max(df_s,0)",
            "weights": {"w1": float(w1), "w2": float(w2), "w3": float(w3), "w4": float(w4)},
        }

    sorted_arms = sorted(finite_arms.items(), key=lambda kv: kv[1])
    reference_arm = str(sorted_arms[0][0])
    arm_margin = float(sorted_arms[1][1] - sorted_arms[0][1]) if len(sorted_arms) > 1 else 0.0
    arm_top2 = [
        {"arm": str(name), "score": float(score)}
        for name, score in sorted_arms[:2]
    ]

    ref_indices = arms_by_name.get(reference_arm, [])
    ref_scores: List[Tuple[int, float]] = [
        (idx, float(chamber_scores_idx[idx]))
        for idx in ref_indices
        if idx in chamber_scores_idx
    ]
    if not ref_scores:
        return {
            "blank_idx_chromaticity": None,
            "blank_status_chromaticity": "unresolved",
            "blank_confidence_chromaticity": 0.0,
            "blank_margin": 0.0,
            "arm_margin": arm_margin,
            "reference_arm": reference_arm,
            "arm_scores": arm_scores,
            "reference_arm_chamber_scores": arm_chamber_scores.get(reference_arm, {}),
            "chamber_scores_by_index": chamber_scores_idx,
            "chamber_features_by_index": chamber_features_by_idx,
            "arm_margin_thr": float(arm_margin_thr),
            "blank_margin_thr": float(blank_margin_thr),
            "clip_quantile": q,
            "clip_rg": clip_rg,
            "clip_re": clip_re,
            "arm_top2": arm_top2,
            "blank_top2": [],
            "mode": "chromaticity",
            "reasons": ["reference_arm_chambers_unavailable"],
            "score_method": "w1*abs(df_r) + w2*abs(df_rg) + w3*max(df_re,0) + w4*max(df_s,0)",
            "weights": {"w1": float(w1), "w2": float(w2), "w3": float(w3), "w4": float(w4)},
        }

    ref_sorted = sorted(ref_scores, key=lambda x: x[1])
    blank_idx = int(ref_sorted[0][0])
    blank_margin = float(ref_sorted[1][1] - ref_sorted[0][1]) if len(ref_sorted) > 1 else 0.0
    blank_confidence = (
        float(np.clip(blank_margin / (abs(ref_sorted[1][1]) + 1e-6), 0.0, 1.0))
        if len(ref_sorted) > 1
        else 0.0
    )
    blank_top2 = [
        {
            "template_index": int(idx),
            "template_id": str(template_ids[idx]) if idx < len(template_ids) else str(idx),
            "score": float(score),
        }
        for idx, score in ref_sorted[:2]
    ]

    status = "confirmed"
    reasons: List[str] = []
    if arm_margin < float(arm_margin_thr):
        status = "unresolved"
        reasons.append(f"arm_margin<{arm_margin_thr:.3f}")
    if blank_margin < float(blank_margin_thr):
        status = "unresolved"
        reasons.append(f"blank_margin<{blank_margin_thr:.3f}")

    return {
        "blank_idx_chromaticity": blank_idx,
        "blank_status_chromaticity": status,
        "blank_confidence_chromaticity": blank_confidence,
        "blank_margin": blank_margin,
        "arm_margin": arm_margin,
        "reference_arm": reference_arm,
        "arm_scores": arm_scores,
        "reference_arm_chamber_scores": arm_chamber_scores.get(reference_arm, {}),
        "chamber_scores_by_index": chamber_scores_idx,
        "chamber_features_by_index": chamber_features_by_idx,
        "arm_margin_thr": float(arm_margin_thr),
        "blank_margin_thr": float(blank_margin_thr),
        "clip_quantile": q,
        "clip_rg": clip_rg,
        "clip_re": clip_re,
        "arm_top2": arm_top2,
        "blank_top2": blank_top2,
        "mode": "chromaticity",
        "reasons": reasons,
        "score_method": "w1*abs(df_r) + w2*abs(df_rg) + w3*max(df_re,0) + w4*max(df_s,0)",
        "weights": {"w1": float(w1), "w2": float(w2), "w3": float(w3), "w4": float(w4)},
    }


def _build_relaxed_post_adaptive_config(config: Stage1Config) -> AdaptiveDetectionConfig:
    """
    构建后处理失败时的宽松重检参数。
    """
    base = (config.adaptive_detection or AdaptiveDetectionConfig()).model_copy(deep=True)
    # Geo-unlock preset: aggressively increase recall for fallback detection.
    base.coarse_conf = 0.008
    base.fine_conf = 0.008
    base.fine_imgsz = int(max(1536, base.fine_imgsz))
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
    geometry_engine: Optional[CrossGeometryEngine] = None,
    export_geo_even_if_blank_unresolved: Optional[bool] = None,
) -> Tuple[List[ChamberDetection], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    对检测点执行拓扑拟合回填，输出固定12点几何检测及 QC。
    """
    n_det_raw = int(len(source_detections))
    effective_geometry_engine = geometry_engine or CrossGeometryEngine(config.geometry)
    topology_cfg = _build_internal_topology_config(config)
    topo_user_cfg = config.topology or TopologyConfig()
    success_mode = str(getattr(topo_user_cfg, "success_mode", "geo_only")).strip().lower()
    if success_mode not in {"geo_only", "geo_and_semantic"}:
        success_mode = "geo_only"
    export_geo = (
        bool(export_geo_even_if_blank_unresolved)
        if export_geo_even_if_blank_unresolved is not None
        else (success_mode != "geo_and_semantic")
    )
    template_ids, blank_candidate_indices, template_name = _load_template_semantics(topology_cfg.template_path)

    prep = _prepare_postprocess_detections(
        source_detections=source_detections,
        roi_bbox=roi_bbox,
        geometry_engine=effective_geometry_engine,
    )
    detections_clean = prep["detections_clean"]
    dedup_meta = prep["dedup_meta"]
    n_det = int(prep["n_det_dedup"])
    pitch_px = float(prep["pitch_final"])
    pitch_method = str(prep["pitch_method"])
    pitch_raw = float(prep["pitch_raw"])
    pitch_filtered = float(prep["pitch_filtered"])
    pitch_guard_triggered = bool(prep["pitch_guard_triggered"])
    det_cleanup_clusters = list(prep["clusters"])

    match_thresh_px = float(max(6.0, ALPHA_MATCH * pitch_px))
    inlier_thresh_px = float(max(4.0, ALPHA_INLIER * pitch_px))
    topology_cfg.ransac_threshold = inlier_thresh_px

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
            source_detections=detections_clean,
            topology_cfg=topology_cfg,
            fit_success=False,
            transform_type="none",
            pitch_px=pitch_px,
            pitch_estimation_method=pitch_method,
            match_thresh_px=match_thresh_px,
            inlier_thresh_px=inlier_thresh_px,
            assignment_method="not_run",
            n_inliers=0,
            rmse_px=None,
            scale=None,
            rotation_deg=None,
            coverage_arms=0.0,
            inlier_indices=[],
            fitted_centers=None,
            visibility=None,
            match_map={},
            match_dists={},
            blank_scores={},
            blank_indices=[],
            blank_confidence=0.0,
            roi_bbox=roi_bbox,
            failure_reason=reason,
            detection_params=detection_params,
            qc={
                "source": source_tag,
                "n_det": int(n_det),
                "n_det_raw": int(n_det_raw),
                "n_det_dedup": int(n_det),
                "pitch_raw": None if not np.isfinite(pitch_raw) else float(pitch_raw),
                "pitch_filtered": None if not np.isfinite(pitch_filtered) else float(pitch_filtered),
                "pitch_final": float(pitch_px),
                "pitch_guard_triggered": bool(pitch_guard_triggered),
                "clusters": det_cleanup_clusters,
                "fit_success": False,
                "fit_success_raw": False,
            },
        )
        raise TopologyPostprocessError(reason, debug_payload=debug_payload)

    fitter = TopologyFitter(topology_cfg)
    template_points = np.asarray(fitter.template, dtype=np.float32)[:12]

    detected_centers = np.array([d.center for d in detections_clean], dtype=np.float32)
    fitting_result = fitter.fit_and_fill(
        detected_centers=detected_centers,
        image_shape=raw_image.shape[:2],
        image=None
    )

    fitted_centers = np.asarray(fitting_result.fitted_centers, dtype=np.float32)[:12]
    strict_match_map, strict_match_dists, strict_assignment_method = _match_template_to_detections_hungarian(
        fitted_centers=fitted_centers,
        source_detections=detections_clean,
        match_thresh_px=match_thresh_px,
    )
    match_map = dict(strict_match_map)
    match_dists = dict(strict_match_dists)
    assignment_method = str(strict_assignment_method)
    matched_count_strict = int(len(strict_match_map))
    matched_count_relaxed = int(matched_count_strict)
    relaxed_match_used = False
    relaxed_assignment_method: Optional[str] = None
    match_thresh_relaxed_px = float(max(
        match_thresh_px,
        float(POSTPROCESS_RECOVERY_CFG["relaxed_match_thresh_ratio"]) * pitch_px,
    ))
    match_thresh_used_px = float(match_thresh_px)
    if (
        n_det >= int(POSTPROCESS_RECOVERY_CFG["relaxed_match_min_det"])
        and matched_count_strict < int(POSTPROCESS_RECOVERY_CFG["relaxed_match_trigger_count"])
    ):
        relaxed_map, relaxed_dists, relaxed_assignment_method = _match_template_to_detections_hungarian(
            fitted_centers=fitted_centers,
            source_detections=detections_clean,
            match_thresh_px=match_thresh_relaxed_px,
        )
        matched_count_relaxed = int(len(relaxed_map))
        if matched_count_relaxed > matched_count_strict:
            match_map = dict(relaxed_map)
            match_dists = dict(relaxed_dists)
            assignment_method = f"{strict_assignment_method}+relaxed"
            match_thresh_used_px = float(match_thresh_relaxed_px)
            relaxed_match_used = True

    matched_count = int(len(match_map))
    pure_filled_count = int(max(0, 12 - matched_count))
    pure_filled_ratio = float(pure_filled_count / 12.0)
    matched_count_per_arm = {
        arm: int(sum(1 for idx in idxs if idx in match_map))
        for arm, idxs in ARM_NAME_TO_INDICES.items()
    }
    matched_arms = int(sum(1 for _, c in matched_count_per_arm.items() if c > 0))
    matched_coverage_arms = float(matched_arms / len(ARM_NAME_TO_INDICES))
    min_matched_count = int(max(
        POST_QC_MIN_MATCHED_COUNT_BASE,
        np.ceil(min_topology_detections * 0.6)
    ))

    reject_det_ratio_used = float(DET_REJECT_RATIO)
    if relaxed_match_used:
        reject_det_ratio_used = float(max(
            reject_det_ratio_used,
            float(POSTPROCESS_RECOVERY_CFG["relaxed_reject_det_ratio"]),
        ))

    final_points_raw, point_source, det_fit_distance_px, final_stats = _build_final_points_det_priority(
        fitted_centers=fitted_centers,
        source_detections=detections_clean,
        match_map=match_map,
        pitch_px=pitch_px,
        reject_det_ratio=reject_det_ratio_used,
    )
    det_used_count = int(sum(1 for s in point_source if s in {"det", "det_mid"}))
    fill_count = int(sum(1 for s in point_source if s == "filled"))
    reject_det_count = int(final_stats["reject_det_count"])
    fill_ratio = float(fill_count / max(1, len(point_source)))
    used_real_points = int(det_used_count)

    rotation_deg, scale = _compute_rotation_scale_from_matrix(fitting_result.transform_matrix)
    coverage_arms_raw = _compute_coverage_arms(fitting_result.detected_mask)
    coverage_arms = matched_coverage_arms
    n_inliers = int(len(fitting_result.inlier_indices))
    rmse_px = float(fitting_result.reprojection_error)

    canonical_points = _compute_canonical_points_from_template(
        template_points=template_points,
        canvas_size=effective_geometry_engine.config.canvas_size,
    )
    transform_summary = _choose_transform_model(
        raw_points=final_points_raw,
        canonical_points=canonical_points,
        point_sources=point_source,
    )
    transform_raw_to_canonical = transform_summary.get("matrix")
    transform_raw_to_canonical_type = str(transform_summary.get("model_chosen", "none"))
    transform_score_sim = float(transform_summary.get("score_sim", float("inf")))
    transform_score_affine = float(transform_summary.get("score_affine", float("inf")))
    transform_score_h = float(transform_summary.get("score_h", float("inf")))
    transform_inliers_sim = int(transform_summary.get("inliers_count_sim", 0))
    transform_inliers_affine = int(transform_summary.get("inliers_count_affine", 0))
    transform_inliers_h = int(transform_summary.get("inliers_count_h", 0))
    transform_reproj_median = float(transform_summary.get("reproj_median", float("inf")))
    transform_reproj_mean = float(transform_summary.get("reproj_mean", float("inf")))
    transform_reproj_max = float(transform_summary.get("reproj_max", float("inf")))
    transform_real_indices = [int(i) for i in transform_summary.get("real_indices", [])]
    transform_used_filled_points = int(transform_summary.get("used_filled_points_for_transform", 0))

    # raw -> canonical 一致性检查（用于证明几何对齐与模板一致）
    reproj_median_px = float("inf")
    reproj_mean_px = float("inf")
    reproj_max_px = float("inf")
    projected_raw = final_points_raw.copy()
    canonical_warp: Optional[np.ndarray] = None
    if transform_raw_to_canonical is not None:
        model = str(transform_raw_to_canonical_type)
        if model == "homography":
            H = np.asarray(transform_raw_to_canonical, dtype=np.float32)
            canonical_warp = cv2.warpPerspective(
                raw_image,
                H,
                (effective_geometry_engine.config.canvas_size, effective_geometry_engine.config.canvas_size),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            try:
                H_inv = np.linalg.inv(H)
                projected_raw = _apply_transform_points(canonical_points, H_inv, "homography")
            except Exception:
                projected_raw = final_points_raw.copy()
        else:
            M = np.asarray(transform_raw_to_canonical, dtype=np.float32)
            canonical_warp = cv2.warpAffine(
                raw_image,
                M,
                (effective_geometry_engine.config.canvas_size, effective_geometry_engine.config.canvas_size),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            inv_m = cv2.invertAffineTransform(M)
            ones = np.ones((canonical_points.shape[0], 1), dtype=np.float32)
            canonical_h = np.hstack([canonical_points.astype(np.float32), ones])
            projected_raw = (inv_m @ canonical_h.T).T
        reproj = np.linalg.norm(projected_raw - final_points_raw.astype(np.float32), axis=1)
        reproj_median_px = float(np.median(reproj))
        reproj_mean_px = float(np.mean(reproj))
        reproj_max_px = float(np.max(reproj))
    if np.isfinite(transform_reproj_median):
        reproj_median_px = float(transform_reproj_median)
    if np.isfinite(transform_reproj_mean):
        reproj_mean_px = float(transform_reproj_mean)
    if np.isfinite(transform_reproj_max):
        reproj_max_px = float(transform_reproj_max)

    indexed_points: List[Dict[str, Any]] = []
    for idx in range(min(12, final_points_raw.shape[0])):
        det_idx = int(match_map[idx]) if idx in match_map else -1
        cluster_id = None
        ref_det_id = None
        if 0 <= det_idx < len(dedup_meta):
            cluster_id = int(dedup_meta[det_idx]["cluster_id"])
            ref_det_id = int(dedup_meta[det_idx]["representative_idx"])
        indexed_points.append(
            {
                "chamber_index": int(idx),
                "chamber_id": str(template_ids[idx]) if idx < len(template_ids) else str(idx),
                "xy": [float(final_points_raw[idx, 0]), float(final_points_raw[idx, 1])],
                "source": str(point_source[idx]) if idx < len(point_source) else "filled",
                "det_id": None if det_idx < 0 else int(det_idx),
                "cluster_id": cluster_id,
                "ref_det_id": ref_det_id,
            }
        )

    geometry_suspect = bool(
        (np.isfinite(reproj_median_px) and reproj_median_px > float(POSTPROCESS_RECOVERY_CFG["fallback_reproj_ratio"]) * pitch_px)
        or used_real_points < int(POSTPROCESS_RECOVERY_CFG["fallback_min_real_points"])
        or fill_ratio > float(POSTPROCESS_RECOVERY_CFG["fallback_max_fill_ratio"])
        or relaxed_match_used
    )
    slice_mode = "det_fallback" if geometry_suspect else "canonical"

    blank_mode = str(getattr(topo_user_cfg, "blank_mode", "chromaticity")).strip().lower()
    if blank_mode == "color_v2":
        blank_mode = "chromaticity"
    if blank_mode not in {"color", "brightness", "chromaticity"}:
        blank_mode = "chromaticity"
    blank_w1 = float(getattr(topo_user_cfg, "blank_w1", 1.0))
    blank_w2 = float(getattr(topo_user_cfg, "blank_w2", 1.0))
    blank_w3 = float(getattr(topo_user_cfg, "blank_w3", 0.5))
    arm_margin_thr = float(getattr(topo_user_cfg, "blank_arm_margin_thr", 0.05))
    blank_margin_thr = float(getattr(topo_user_cfg, "blank_margin_thr", 0.05))
    blank_chroma_w1 = float(getattr(topo_user_cfg, "blank_chroma_w1", 1.0))
    blank_chroma_w2 = float(getattr(topo_user_cfg, "blank_chroma_w2", 1.0))
    blank_chroma_w3 = float(getattr(topo_user_cfg, "blank_chroma_w3", 0.5))
    blank_chroma_w4 = float(getattr(topo_user_cfg, "blank_chroma_w4", 0.5))
    blank_chroma_arm_margin_thr = float(getattr(topo_user_cfg, "blank_chroma_arm_margin_thr", 0.05))
    blank_chroma_blank_margin_thr = float(getattr(topo_user_cfg, "blank_chroma_blank_margin_thr", 0.02))
    blank_chroma_clip_quantile = float(getattr(topo_user_cfg, "blank_chroma_clip_quantile", 0.99))
    blank_chroma_clip_rg_min = float(getattr(topo_user_cfg, "blank_chroma_clip_rg_min", 0.15))
    blank_chroma_clip_re_min = float(getattr(topo_user_cfg, "blank_chroma_clip_re_min", 5.0))

    candidate_indices = (
        [int(x) for x in blank_candidate_indices]
        if blank_candidate_indices
        else list(OUTERMOST_TEMPLATE_INDICES)
    )
    candidate_indices = [
        int(i) for i in candidate_indices
        if int(i) in set(OUTERMOST_TEMPLATE_INDICES) and 0 <= int(i) < 12
    ]
    if not candidate_indices:
        candidate_indices = list(OUTERMOST_TEMPLATE_INDICES)

    blank_indices: List[int] = []
    blank_scores: Dict[int, float] = {}
    blank_confidence = 0.0
    blank_score_spread = 0.0
    blank_features: Dict[int, Dict[str, float]] = {}
    blank_status = "unresolved"
    blank_fail_reasons: List[str] = []
    blank_mode_selected = blank_mode
    reference_arm: Optional[str] = None
    reference_arm_pred: Optional[str] = None
    arm_scores: Dict[str, float] = {}
    reference_arm_chamber_scores: Dict[str, float] = {}
    blank_chamber_scores_idx: Dict[int, float] = {}
    blank_chamber_scores_by_template: Dict[str, float] = {}
    blank_chamber_features_by_template: Dict[str, Dict[str, float]] = {}
    blank_id_old: Optional[int] = None
    blank_id_color: Optional[int] = None
    blank_id_chromaticity: Optional[int] = None
    blank_status_color = "unresolved"
    blank_confidence_color = 0.0
    blank_margin_color = 0.0
    arm_margin_color = 0.0
    blank_status_chromaticity = "unresolved"
    blank_confidence_chromaticity = 0.0
    blank_margin_chromaticity = 0.0
    arm_margin_chromaticity = 0.0
    blank_id_pred: Optional[int] = None
    blank_status_pred = "unresolved"
    blank_confidence_pred = 0.0
    blank_score_pred = float("nan")
    blank_margin_pred = 0.0
    arm_margin_pred = 0.0
    blank_unresolved = True
    blank_is_outermost = False
    selected_arm_margin_thr = float(arm_margin_thr)
    selected_blank_margin_thr = float(blank_margin_thr)
    arm_top2: List[Dict[str, Any]] = []
    blank_top2: List[Dict[str, Any]] = []
    score_method = "0.45*V_low + 0.25*|dV| + 0.20*S_med + 0.40*sat_ratio*255"
    score_weights: Dict[str, float] = {}

    if canonical_warp is None:
        blank_fail_reasons.append("blank_canonical_warp_failed")
    else:
        canonical_pitch = _estimate_pitch_from_points(canonical_points)
        if not np.isfinite(canonical_pitch):
            canonical_pitch = float(max(8.0, topology_cfg.template_scale))
        chamber_radius_px = float(max(4.0, canonical_pitch * 0.23))

        (
            old_indices,
            old_scores,
            old_confidence,
            old_spread,
            old_features,
        ) = _compute_blank_brightness_outermost(
            canonical_image=canonical_warp,
            canonical_centers=canonical_points,
            candidate_indices=candidate_indices,
            chamber_radius_px=chamber_radius_px,
        )
        blank_id_old = int(old_indices[0]) if old_indices else None

        if blank_mode == "brightness":
            blank_indices = [blank_id_old] if blank_id_old is not None else []
            blank_scores = dict(old_scores)
            blank_confidence = float(old_confidence)
            blank_score_spread = float(old_spread)
            blank_features = {
                int(k): {kk: float(vv) for kk, vv in feat.items()}
                for k, feat in old_features.items()
            }
            selected_blank_margin_thr = float(BLANK_MARGIN_THRESHOLD)
            blank_id_pred = blank_id_old
            blank_status_pred = "confirmed_detected" if blank_id_pred is not None else "unresolved"
            blank_confidence_pred = float(blank_confidence)
            blank_score_pred = (
                float(blank_scores.get(int(blank_id_pred), float("nan")))
                if blank_id_pred is not None
                else float("nan")
            )
            blank_margin_pred = float(blank_score_spread)
            arm_margin_pred = 0.0
            if blank_id_pred is not None:
                reference_arm_pred = ["Up", "Right", "Down", "Left"][max(0, min(3, int(blank_id_pred) // 3))]
            blank_unresolved = bool(blank_status_pred == "unresolved")
            blank_is_outermost = bool(blank_id_pred in OUTERMOST_TEMPLATE_INDICES) if blank_id_pred is not None else False
            score_method = "0.45*V_low + 0.25*|dV| + 0.20*S_med + 0.40*sat_ratio*255"
            score_weights = {}
        elif blank_mode == "color":
            color_result = _compute_blank_color_mode(
                canonical_image=canonical_warp,
                canonical_centers=canonical_points,
                template_ids=template_ids,
                w1=blank_w1,
                w2=blank_w2,
                w3=blank_w3,
                arm_margin_thr=arm_margin_thr,
                blank_margin_thr=blank_margin_thr,
                chamber_radius_px=chamber_radius_px,
            )
            blank_id_color = (
                int(color_result["blank_idx_color"])
                if color_result.get("blank_idx_color") is not None
                else None
            )
            blank_status_color = str(color_result.get("blank_status_color", "unresolved"))
            blank_confidence_color = float(color_result.get("blank_confidence_color", 0.0))
            blank_margin_color = float(color_result.get("blank_margin", 0.0))
            arm_margin_color = float(color_result.get("arm_margin", 0.0))
            selected_arm_margin_thr = float(color_result.get("arm_margin_thr", arm_margin_thr))
            selected_blank_margin_thr = float(color_result.get("blank_margin_thr", blank_margin_thr))
            reference_arm = color_result.get("reference_arm")
            arm_scores = {
                str(k): (float(v) if np.isfinite(v) else float("inf"))
                for k, v in (color_result.get("arm_scores", {}) or {}).items()
            }
            reference_arm_chamber_scores = {
                str(k): float(v)
                for k, v in (color_result.get("reference_arm_chamber_scores", {}) or {}).items()
            }
            blank_chamber_scores_idx = {
                int(k): float(v)
                for k, v in (color_result.get("chamber_scores_by_index", {}) or {}).items()
            }
            color_features_idx = color_result.get("chamber_features_by_index", {}) or {}
            blank_chamber_scores_by_template = {
                str(template_ids[idx]): float(score)
                for idx, score in blank_chamber_scores_idx.items()
                if 0 <= idx < len(template_ids)
            }
            blank_chamber_features_by_template = {
                str(template_ids[int(idx)]): {kk: float(vv) for kk, vv in feat.items()}
                for idx, feat in color_features_idx.items()
                if 0 <= int(idx) < len(template_ids)
            }
            blank_indices = [blank_id_color] if blank_id_color is not None else []
            blank_scores = dict(blank_chamber_scores_idx)
            blank_confidence = float(blank_confidence_color)
            blank_score_spread = float(blank_margin_color)
            blank_features = {
                int(k): {kk: float(vv) for kk, vv in feat.items()}
                for k, feat in color_features_idx.items()
            }
            arm_top2 = []
            blank_top2 = []
            score_method = "w1*abs(delta_f_rg) + w2*delta_f_red + w3*delta_f_s"
            score_weights = {"w1": float(blank_w1), "w2": float(blank_w2), "w3": float(blank_w3)}
            arm_sorted = sorted(
                [(str(k), float(v)) for k, v in arm_scores.items() if np.isfinite(float(v))],
                key=lambda kv: kv[1],
            )
            arm_top2 = [{"arm": name, "score": score} for name, score in arm_sorted[:2]]
            ref_sorted = sorted(
                [(str(tid), float(score)) for tid, score in reference_arm_chamber_scores.items()],
                key=lambda kv: kv[1],
            )
            blank_top2 = []
            for tid, score in ref_sorted[:2]:
                try:
                    idx = int(template_ids.index(tid))
                except Exception:
                    idx = -1
                blank_top2.append(
                    {
                        "template_index": idx,
                        "template_id": tid,
                        "score": float(score),
                    }
                )
            blank_id_pred = blank_id_color
            blank_status_pred = blank_status_color
            blank_confidence_pred = float(blank_confidence_color)
            blank_score_pred = (
                float(blank_chamber_scores_idx.get(int(blank_id_pred), float("nan")))
                if blank_id_pred is not None
                else float("nan")
            )
            blank_margin_pred = float(blank_margin_color)
            arm_margin_pred = float(arm_margin_color)
            reference_arm_pred = reference_arm
            blank_unresolved = bool(blank_status_color == "unresolved")
            blank_is_outermost = bool(blank_id_pred in OUTERMOST_TEMPLATE_INDICES) if blank_id_pred is not None else False
            if blank_status_color == "unresolved":
                blank_fail_reasons.extend(list(color_result.get("reasons", [])))
        else:
            chroma_result = _compute_blank_chromaticity_mode(
                canonical_image=canonical_warp,
                canonical_centers=canonical_points,
                template_ids=template_ids,
                w1=blank_chroma_w1,
                w2=blank_chroma_w2,
                w3=blank_chroma_w3,
                w4=blank_chroma_w4,
                arm_margin_thr=blank_chroma_arm_margin_thr,
                blank_margin_thr=blank_chroma_blank_margin_thr,
                clip_quantile=blank_chroma_clip_quantile,
                clip_rg_min=blank_chroma_clip_rg_min,
                clip_re_min=blank_chroma_clip_re_min,
                chamber_radius_px=chamber_radius_px,
            )
            blank_id_chromaticity = (
                int(chroma_result["blank_idx_chromaticity"])
                if chroma_result.get("blank_idx_chromaticity") is not None
                else None
            )
            blank_status_chromaticity = str(chroma_result.get("blank_status_chromaticity", "unresolved"))
            blank_confidence_chromaticity = float(chroma_result.get("blank_confidence_chromaticity", 0.0))
            blank_margin_chromaticity = float(chroma_result.get("blank_margin", 0.0))
            arm_margin_chromaticity = float(chroma_result.get("arm_margin", 0.0))
            selected_arm_margin_thr = float(chroma_result.get("arm_margin_thr", blank_chroma_arm_margin_thr))
            selected_blank_margin_thr = float(chroma_result.get("blank_margin_thr", blank_chroma_blank_margin_thr))
            reference_arm = chroma_result.get("reference_arm")
            arm_scores = {
                str(k): (float(v) if np.isfinite(v) else float("inf"))
                for k, v in (chroma_result.get("arm_scores", {}) or {}).items()
            }
            reference_arm_chamber_scores = {
                str(k): float(v)
                for k, v in (chroma_result.get("reference_arm_chamber_scores", {}) or {}).items()
            }
            blank_chamber_scores_idx = {
                int(k): float(v)
                for k, v in (chroma_result.get("chamber_scores_by_index", {}) or {}).items()
            }
            chroma_features_idx = chroma_result.get("chamber_features_by_index", {}) or {}
            blank_chamber_scores_by_template = {
                str(template_ids[idx]): float(score)
                for idx, score in blank_chamber_scores_idx.items()
                if 0 <= idx < len(template_ids)
            }
            blank_chamber_features_by_template = {
                str(template_ids[int(idx)]): {kk: float(vv) for kk, vv in feat.items()}
                for idx, feat in chroma_features_idx.items()
                if 0 <= int(idx) < len(template_ids)
            }
            arm_top2 = list(chroma_result.get("arm_top2", []) or [])
            blank_top2 = list(chroma_result.get("blank_top2", []) or [])
            blank_indices = [blank_id_chromaticity] if blank_id_chromaticity is not None else []
            blank_scores = dict(blank_chamber_scores_idx)
            blank_confidence = float(blank_confidence_chromaticity)
            blank_score_spread = float(blank_margin_chromaticity)
            blank_features = {
                int(k): {kk: float(vv) for kk, vv in feat.items()}
                for k, feat in chroma_features_idx.items()
            }
            score_method = str(chroma_result.get("score_method", "w1*abs(df_r)+w2*abs(df_rg)+w3*max(df_re,0)+w4*max(df_s,0)"))
            score_weights = {
                str(k): float(v)
                for k, v in (chroma_result.get("weights", {}) or {}).items()
            }
            blank_id_pred = blank_id_chromaticity
            blank_status_pred = blank_status_chromaticity
            blank_confidence_pred = float(blank_confidence_chromaticity)
            blank_score_pred = (
                float(blank_chamber_scores_idx.get(int(blank_id_pred), float("nan")))
                if blank_id_pred is not None
                else float("nan")
            )
            blank_margin_pred = float(blank_margin_chromaticity)
            arm_margin_pred = float(arm_margin_chromaticity)
            reference_arm_pred = reference_arm
            blank_unresolved = bool(blank_status_chromaticity == "unresolved")
            blank_is_outermost = bool(blank_id_pred in OUTERMOST_TEMPLATE_INDICES) if blank_id_pred is not None else False
            if blank_status_chromaticity == "unresolved":
                blank_fail_reasons.extend(list(chroma_result.get("reasons", [])))

        if not blank_indices:
            blank_fail_reasons.append("blank_candidates_insufficient")

    forced_blank_idx, allowed_blank_candidates, forced_blank_is_outer = _select_blank_with_outermost_constraint(
        blank_indices=blank_indices,
        blank_scores=blank_scores,
        reference_arm=reference_arm_pred if reference_arm_pred is not None else reference_arm,
        template_ids=template_ids,
        candidate_indices=candidate_indices,
    )
    blank_indices = [int(forced_blank_idx)] if forced_blank_idx is not None else []
    blank_is_outermost = bool(forced_blank_is_outer)
    if forced_blank_idx is None:
        blank_fail_reasons.append("blank_outermost_unresolved")
    if forced_blank_idx is not None and forced_blank_idx not in allowed_blank_candidates:
        blank_fail_reasons.append("blank_not_in_allowed_outermost_candidates")

    reproj_thresh = float(POST_QC_MAX_GEOMETRY_REPROJ_RATIO * pitch_px)
    blank_idx = int(blank_indices[0]) if blank_indices else None
    blank_source = point_source[blank_idx] if blank_idx is not None and blank_idx < len(point_source) else "none"
    blank_from_non_det = blank_source not in ("det", "det_mid")
    blank_spread_gate = float(
        selected_blank_margin_thr
        if blank_mode in {"color", "chromaticity"}
        else BLANK_MARGIN_THRESHOLD
    )

    if blank_idx is None:
        blank_status = "unresolved"
    elif blank_score_spread < blank_spread_gate:
        blank_status = "unresolved"
        blank_fail_reasons.append(f"blank_margin<{blank_spread_gate:.3f}")
    elif blank_from_non_det:
        strict_margin_thr = float(BLANK_FILLED_MARGIN_SCALE * blank_spread_gate)
        cond_matched = bool(
            matched_count >= BLANK_FILLED_MIN_MATCHED_COUNT
            or det_used_count >= BLANK_FILLED_MIN_DET_USED_COUNT
        )
        cond_reproj = bool(np.isfinite(reproj_mean_px) and reproj_mean_px <= reproj_thresh)
        cond_margin = bool(blank_score_spread >= strict_margin_thr)
        if cond_matched and cond_reproj and cond_margin:
            blank_status = "confirmed_filled"
        else:
            blank_status = "unresolved"
            if not cond_matched:
                blank_fail_reasons.append("blank_filled_matched_count_low")
            if not cond_reproj:
                blank_fail_reasons.append("blank_filled_reprojection_high")
            if not cond_margin:
                blank_fail_reasons.append("blank_filled_margin_low")
    else:
        blank_status = "confirmed_detected"

    blank_is_detected = bool(blank_status == "confirmed_detected")
    blank_id_pred = blank_idx
    blank_status_pred = blank_status
    blank_confidence_pred = float(blank_confidence)
    blank_score_pred = (
        float(blank_scores.get(int(blank_idx), float("nan")))
        if blank_idx is not None
        else float("nan")
    )
    blank_margin_pred = float(blank_score_spread)
    blank_unresolved = bool(blank_status == "unresolved")
    blank_is_outermost = bool(blank_idx in OUTERMOST_TEMPLATE_INDICES) if blank_idx is not None else False
    blank_valid = bool(blank_idx is not None and blank_is_outermost)
    if blank_idx is not None and not blank_is_outermost:
        blank_fail_reasons.append("blank_not_outermost")
    if reference_arm_pred is None:
        reference_arm_pred = reference_arm
    if reference_arm_pred is None and blank_idx is not None:
        reference_arm_pred = ["Up", "Right", "Down", "Left"][max(0, min(3, int(blank_idx) // 3))]
    reference_arm = reference_arm_pred

    topology_success_geo = bool(
        fitting_result.fit_success
        and n_inliers >= int(topology_cfg.min_inliers)
        and np.isfinite(rmse_px)
        and coverage_arms >= 0.25
        and scale > 0.0
        and matched_count >= min_matched_count
        and matched_coverage_arms >= POST_QC_MIN_MATCHED_COVERAGE_ARMS
        and pure_filled_ratio <= POST_QC_MAX_PURE_FILLED_RATIO
        and transform_raw_to_canonical is not None
        and transform_raw_to_canonical_type in {"similarity", "affine", "homography"}
    )
    semantic_ready = bool(
        reference_arm_pred is not None
        and blank_id_pred is not None
        and blank_confidence >= POST_QC_MIN_BLANK_CONFIDENCE
        and blank_score_spread >= blank_spread_gate
        and blank_status != "unresolved"
        and blank_valid
        and (blank_is_detected if POST_QC_REQUIRE_BLANK_DETECTED else True)
    )
    fit_success_qc = bool(topology_success_geo)

    qc_fail_reasons_geo: List[str] = []
    if not fitting_result.fit_success:
        qc_fail_reasons_geo.append("fit_success_raw=false")
    if n_inliers < int(topology_cfg.min_inliers):
        qc_fail_reasons_geo.append(f"n_inliers<{int(topology_cfg.min_inliers)}")
    if not np.isfinite(rmse_px):
        qc_fail_reasons_geo.append("rmse_non_finite")
    if coverage_arms < 0.25:
        qc_fail_reasons_geo.append("coverage_arms<0.25")
    if scale <= 0.0:
        qc_fail_reasons_geo.append("scale<=0")
    if matched_count < min_matched_count:
        qc_fail_reasons_geo.append(f"matched_count<{min_matched_count}")
    if matched_coverage_arms < POST_QC_MIN_MATCHED_COVERAGE_ARMS:
        qc_fail_reasons_geo.append(f"matched_coverage_arms<{POST_QC_MIN_MATCHED_COVERAGE_ARMS:.2f}")
    if pure_filled_ratio > POST_QC_MAX_PURE_FILLED_RATIO:
        qc_fail_reasons_geo.append(f"pure_filled_ratio>{POST_QC_MAX_PURE_FILLED_RATIO:.2f}")
    if transform_raw_to_canonical is None:
        qc_fail_reasons_geo.append("transform_unavailable")
    if transform_raw_to_canonical_type not in {"similarity", "affine", "homography"}:
        qc_fail_reasons_geo.append("transform_model_invalid")
    semantic_fail_reasons: List[str] = []
    if reference_arm_pred is None:
        semantic_fail_reasons.append("reference_arm_unresolved")
    if blank_id_pred is None:
        semantic_fail_reasons.append("blank_id_unresolved")
    if blank_confidence < POST_QC_MIN_BLANK_CONFIDENCE:
        semantic_fail_reasons.append(f"blank_confidence<{POST_QC_MIN_BLANK_CONFIDENCE:.3f}")
    if blank_score_spread < blank_spread_gate:
        semantic_fail_reasons.append(f"blank_score_spread<{blank_spread_gate:.3f}")
    if blank_status == "unresolved":
        semantic_fail_reasons.append("blank_unresolved")
        semantic_fail_reasons.extend(blank_fail_reasons)
    if not blank_valid:
        semantic_fail_reasons.append("blank_invalid")
    if POST_QC_REQUIRE_BLANK_DETECTED and not blank_is_detected:
        semantic_fail_reasons.append("blank_not_detected")
    qc_fail_reasons: List[str] = []
    qc_fail_reasons.extend(qc_fail_reasons_geo)
    qc_fail_reasons.extend(semantic_fail_reasons)

    blank_idx = int(blank_indices[0]) if blank_indices else None
    (
        semantic_order_clockwise_from_blank,
        semantic_roles_by_template,
        semantic_arm_role_by_template,
    ) = _build_semantic_roles(template_ids, blank_idx)

    qc: Dict[str, Any] = {
        "source": source_tag,
        "used_fallback": bool(used_fallback),
        "n_det_raw": int(n_det_raw),
        "n_det_dedup": int(n_det),
        "clusters": det_cleanup_clusters,
        "n_det": int(n_det),
        "n_inliers": int(n_inliers),
        "n_filled": int(pure_filled_count),
        "pure_filled_count": int(pure_filled_count),
        "pure_filled_ratio": float(pure_filled_ratio),
        "fill_ratio": float(fill_ratio),
        "used_real_points": int(used_real_points),
        "indexed_points": indexed_points,
        "matched_count": int(matched_count),
        "det_used_count": int(det_used_count),
        "fill_count": int(fill_count),
        "reject_det_count": int(reject_det_count),
        "det_fit_dist_mean_px": final_stats.get("det_fit_dist_mean_px"),
        "det_fit_dist_max_px": final_stats.get("det_fit_dist_max_px"),
        "keep_det_thr_px": final_stats.get("keep_det_thr_px"),
        "reject_det_thr_px": final_stats.get("reject_det_thr_px"),
        "reject_det_ratio_used": float(reject_det_ratio_used),
        "matched_count_strict": int(matched_count_strict),
        "matched_count_relaxed": int(matched_count_relaxed),
        "relaxed_match_used": bool(relaxed_match_used),
        "match_thresh_strict_px": float(match_thresh_px),
        "match_thresh_used_px": float(match_thresh_used_px),
        "match_thresh_relaxed_px": float(match_thresh_relaxed_px),
        "assignment_method_strict": str(strict_assignment_method),
        "assignment_method_relaxed": (
            None if relaxed_assignment_method is None else str(relaxed_assignment_method)
        ),
        "matched_count_per_arm": matched_count_per_arm,
        "matched_coverage_arms": float(matched_coverage_arms),
        "blank_is_detected": bool(blank_is_detected),
        "blank_source": blank_source,
        "blank_status": blank_status,
        "blank_mode": blank_mode_selected,
        "blank_id_old": blank_id_old,
        "blank_id_color": blank_id_color,
        "blank_id_chromaticity": blank_id_chromaticity,
        "blank_id_pred": blank_id_pred,
        "blank_status_color": blank_status_color,
        "blank_status_chromaticity": blank_status_chromaticity,
        "blank_status_pred": blank_status_pred,
        "blank_confidence_color": float(blank_confidence_color),
        "blank_confidence_chromaticity": float(blank_confidence_chromaticity),
        "blank_confidence_pred": float(blank_confidence_pred),
        "blank_margin_color": float(blank_margin_color),
        "blank_margin_chromaticity": float(blank_margin_chromaticity),
        "blank_margin_pred": float(blank_margin_pred),
        "arm_margin_color": float(arm_margin_color),
        "arm_margin_chromaticity": float(arm_margin_chromaticity),
        "arm_margin_pred": float(arm_margin_pred),
        "blank_spread_gate": float(blank_spread_gate),
        "blank_arm_margin_thr": float(selected_arm_margin_thr),
        "blank_margin_thr": float(selected_blank_margin_thr),
        "blank_score_method": score_method,
        "blank_score_weights": score_weights,
        "blank_score": float(blank_score_pred) if np.isfinite(blank_score_pred) else None,
        "blank_unresolved": bool(blank_unresolved),
        "blank_is_outermost": bool(blank_is_outermost),
        "blank_valid": bool(blank_valid),
        "blank_selected": (
            str(template_ids[blank_id_pred]) if blank_id_pred is not None and 0 <= int(blank_id_pred) < len(template_ids)
            else None
        ),
        "blank_selected_index": (None if blank_id_pred is None else int(blank_id_pred)),
        "blank_candidates": [
            str(template_ids[idx]) if 0 <= int(idx) < len(template_ids) else str(int(idx))
            for idx in allowed_blank_candidates
        ],
        "blank_candidate_indices": [int(i) for i in allowed_blank_candidates],
        "success_mode": success_mode,
        "export_geo_even_if_blank_unresolved": bool(export_geo),
        "topology_success_geo": bool(topology_success_geo),
        "semantic_ready": bool(semantic_ready),
        "blank_pass": bool(semantic_ready),
        "geo_success": bool(topology_success_geo),
        "reference_arm": reference_arm,
        "reference_arm_pred": reference_arm_pred,
        "arm_scores": {str(k): float(v) for k, v in arm_scores.items()},
        "reference_arm_chamber_scores": {
            str(k): float(v) for k, v in reference_arm_chamber_scores.items()
        },
        "arm_top2": arm_top2,
        "blank_top2": blank_top2,
        "blank_chamber_scores_by_template": {
            str(k): float(v) for k, v in blank_chamber_scores_by_template.items()
        },
        "blank_chamber_features_by_template": blank_chamber_features_by_template,
        "color_chamber_scores_by_template": {
            str(k): float(v) for k, v in blank_chamber_scores_by_template.items()
        },
        "color_chamber_features_by_template": blank_chamber_features_by_template,
        "min_matched_count": int(min_matched_count),
        "max_pure_filled_ratio": float(POST_QC_MAX_PURE_FILLED_RATIO),
        "min_matched_coverage_arms": float(POST_QC_MIN_MATCHED_COVERAGE_ARMS),
        "require_blank_detected": bool(POST_QC_REQUIRE_BLANK_DETECTED),
        "rmse_px": float(rmse_px),
        "inlier_ratio": float(fitting_result.inlier_ratio),
        "coverage_arms": float(coverage_arms),
        "coverage_arms_raw": float(coverage_arms_raw),
        "scale": float(scale),
        "rotation": float(rotation_deg),
        "transform_type": str(transform_raw_to_canonical_type),
        "topology_fit_transform_type": str(fitting_result.transform_type),
        "canonical_transform_type": str(transform_raw_to_canonical_type),
        "model_chosen": str(transform_raw_to_canonical_type),
        "score_sim": float(transform_score_sim) if np.isfinite(transform_score_sim) else None,
        "score_affine": float(transform_score_affine) if np.isfinite(transform_score_affine) else None,
        "score_h": float(transform_score_h) if np.isfinite(transform_score_h) else None,
        "inliers_count_sim": int(transform_inliers_sim),
        "inliers_count_affine": int(transform_inliers_affine),
        "inliers_count_h": int(transform_inliers_h),
        "transform_real_indices": transform_real_indices,
        "transform_used_filled_points": int(transform_used_filled_points),
        "reproj_median": float(reproj_median_px) if np.isfinite(reproj_median_px) else None,
        "reproj_mean": float(reproj_mean_px) if np.isfinite(reproj_mean_px) else None,
        "reproj_max": float(reproj_max_px) if np.isfinite(reproj_max_px) else None,
        "geometry_suspect": bool(geometry_suspect),
        "slice_mode": str(slice_mode),
        "pitch_px": float(pitch_px),
        "pitch_raw": float(pitch_raw) if np.isfinite(pitch_raw) else None,
        "pitch_filtered": float(pitch_filtered) if np.isfinite(pitch_filtered) else None,
        "pitch_final": float(pitch_px),
        "pitch_guard_triggered": bool(pitch_guard_triggered),
        "pitch_estimation_method": str(pitch_method),
        "match_thresh_px": float(match_thresh_used_px),
        "inlier_thresh_px": float(inlier_thresh_px),
        "assignment_method": str(assignment_method),
        "blank_scores": {str(k): float(v) for k, v in blank_scores.items()},
        "blank_confidence": float(blank_confidence),
        "blank_score_spread": float(blank_score_spread),
        "blank_indices": [int(x) for x in blank_indices],
        "blank_features": {
            str(k): {kk: float(vv) for kk, vv in feat.items()}
            for k, feat in blank_features.items()
        },
        "template_name": template_name,
        "template_ids": template_ids,
        "semantic_roles_by_template": semantic_roles_by_template,
        "semantic_arm_role_by_template": semantic_arm_role_by_template,
        "blank_candidate_indices_raw": blank_candidate_indices,
        "reprojection_error_mean_px": float(reproj_mean_px) if np.isfinite(reproj_mean_px) else None,
        "reprojection_error_max_px": float(reproj_max_px) if np.isfinite(reproj_max_px) else None,
        "reprojection_error_median_px": float(reproj_median_px) if np.isfinite(reproj_median_px) else None,
        "reprojection_error_threshold_px": float(reproj_thresh),
        "semantic_order_clockwise_from_blank": semantic_order_clockwise_from_blank,
        "fit_success_raw": bool(fitting_result.fit_success),
        "fit_success": bool(fit_success_qc),
        "qc_fail_reasons_geo": qc_fail_reasons_geo,
        "semantic_fail_reasons": semantic_fail_reasons,
        "qc_fail_reasons": qc_fail_reasons,
    }

    debug_payload = _build_topology_attempt_debug_payload(
        chip_id=chip_id,
        attempt_id=attempt_id,
        status="success",
        raw_image_path=raw_image_path,
        source_tag=source_tag,
        used_fallback=used_fallback,
        min_topology_detections=min_topology_detections,
        source_detections=detections_clean,
        topology_cfg=topology_cfg,
        fit_success=fit_success_qc,
        transform_type=str(transform_raw_to_canonical_type),
        pitch_px=pitch_px,
        pitch_estimation_method=pitch_method,
        match_thresh_px=match_thresh_used_px,
        inlier_thresh_px=inlier_thresh_px,
        assignment_method=assignment_method,
        n_inliers=n_inliers,
        rmse_px=rmse_px,
        scale=scale,
        rotation_deg=rotation_deg,
        coverage_arms=coverage_arms,
        inlier_indices=[int(i) for i in fitting_result.inlier_indices],
        fitted_centers=fitted_centers,
        final_points=final_points_raw,
        point_sources=point_source,
        det_fit_distances=det_fit_distance_px,
        visibility=np.asarray(fitting_result.visibility, dtype=bool),
        match_map=match_map,
        match_dists=match_dists,
        blank_scores=blank_scores,
        blank_indices=blank_indices,
        blank_confidence=blank_confidence,
        blank_status=blank_status,
        blank_features=blank_features,
        blank_score_spread=blank_score_spread,
        blank_candidate_indices=allowed_blank_candidates,
        blank_mode=blank_mode_selected,
        blank_id_old=blank_id_old,
        blank_id_color=blank_id_color,
        blank_id_chromaticity=blank_id_chromaticity,
        blank_status_color=blank_status_color,
        blank_status_chromaticity=blank_status_chromaticity,
        reference_arm=reference_arm,
        reference_arm_pred=reference_arm_pred,
        arm_scores=arm_scores,
        reference_arm_chamber_scores=reference_arm_chamber_scores,
        arm_margin=arm_margin_pred,
        blank_margin=blank_margin_pred,
        chamber_scores_by_template=blank_chamber_scores_by_template,
        chamber_features_by_template=blank_chamber_features_by_template,
        arm_top2=arm_top2,
        blank_top2=blank_top2,
        score_method=score_method,
        score_weights=score_weights,
        template_ids=template_ids,
        semantic_roles_by_template=semantic_roles_by_template,
        semantic_arm_role_by_template=semantic_arm_role_by_template,
        det_priority_stats=final_stats,
        roi_bbox=roi_bbox,
        failure_reason=None,
        detection_params=detection_params,
        qc=qc,
    )

    if not fit_success_qc:
        reason = (
            "postprocess_qc_failed: topology_fit_failed "
            f"(n_det={n_det}, n_inliers={n_inliers}, rmse_px={rmse_px:.2f}, "
            f"coverage_arms={coverage_arms:.2f}, matched_count={matched_count}, "
            f"matched_coverage_arms={matched_coverage_arms:.2f}, "
            f"pure_filled_ratio={pure_filled_ratio:.2f}, blank_status={blank_status}, blank_is_detected={blank_is_detected}, "
            f"reasons={qc_fail_reasons})"
        )
        debug_payload["status"] = "failed"
        debug_payload["failure_reason"] = reason
        raise TopologyPostprocessError(reason, debug_payload=debug_payload)
    if (blank_status == "unresolved" or not blank_indices) and (not export_geo):
        reason = (
            "postprocess_qc_failed: blank_unresolved "
            f"(n_det={n_det}, coverage_arms={coverage_arms:.2f}, reasons={blank_fail_reasons})"
        )
        debug_payload["status"] = "failed"
        debug_payload["failure_reason"] = reason
        raise TopologyPostprocessError(reason, debug_payload=debug_payload)

    adaptive_result = AdaptiveDetectionResult(
        detections=detections_clean,
        roi_bbox=(0, 0, int(raw_image.shape[1]), int(raw_image.shape[0])),
        cluster_score=0.0,
        is_fallback=(source_tag != "json"),
        fitted_centers=final_points_raw,
        visibility=np.asarray(fitting_result.visibility, dtype=bool),
        detected_mask=np.array([ps in ("det", "det_mid") for ps in point_source], dtype=bool),
        dark_chamber_indices=blank_indices,
        inlier_ratio=float(fitting_result.inlier_ratio),
        reprojection_error=float(fitting_result.reprojection_error),
        fit_success=bool(fitting_result.fit_success),
        processing_time=0.0
    )

    geometry_detections = _build_geometry_detections(
        adaptive_result=adaptive_result,
        geometry_engine=effective_geometry_engine,
        class_id_blank=config.yolo.class_id_blank,
        class_id_lit=config.yolo.class_id_lit,
        force_blank_if_missing=False,
        match_map=match_map
    )
    canonical_context = {
        "template_points": template_points.tolist(),
        "fitted_points_raw": fitted_centers.tolist(),
        "final_points_raw": final_points_raw.tolist(),
        "point_source": point_source,
        "det_fit_distance_px": [None if d is None else float(d) for d in det_fit_distance_px],
        "matched_det_indices": [int(x) for x in final_stats.get("matched_det_indices", [])],
        "blank_idx": blank_idx,
        "blank_status": blank_status,
        "blank_mode": blank_mode_selected,
        "blank_id_old": blank_id_old,
        "blank_id_color": blank_id_color,
        "blank_id_chromaticity": blank_id_chromaticity,
        "blank_id_pred": blank_id_pred,
        "blank_status_color": blank_status_color,
        "blank_status_chromaticity": blank_status_chromaticity,
        "blank_status_pred": blank_status_pred,
        "reference_arm": reference_arm,
        "reference_arm_pred": reference_arm_pred,
        "arm_scores": arm_scores,
        "reference_arm_chamber_scores": reference_arm_chamber_scores,
        "blank_margin_color": float(blank_margin_color),
        "arm_margin_color": float(arm_margin_color),
        "blank_margin_chromaticity": float(blank_margin_chromaticity),
        "arm_margin_chromaticity": float(arm_margin_chromaticity),
        "blank_margin_pred": float(blank_margin_pred),
        "arm_margin_pred": float(arm_margin_pred),
        "blank_unresolved": bool(blank_unresolved),
        "blank_is_outermost": bool(blank_is_outermost),
        "blank_score": float(blank_score_pred) if np.isfinite(blank_score_pred) else None,
        "arm_top2": arm_top2,
        "blank_top2": blank_top2,
        "blank_chamber_scores_by_template": blank_chamber_scores_by_template,
        "blank_chamber_features_by_template": blank_chamber_features_by_template,
        "blank_score_method": score_method,
        "blank_score_weights": score_weights,
        "template_ids": template_ids,
        "blank_candidate_indices": [int(i) for i in allowed_blank_candidates],
        "semantic_order_clockwise_from_blank": semantic_order_clockwise_from_blank,
        "semantic_roles_by_template": semantic_roles_by_template,
        "semantic_arm_role_by_template": semantic_arm_role_by_template,
        "transform_raw_to_canonical": (
            np.asarray(transform_raw_to_canonical).tolist()
            if transform_raw_to_canonical is not None
            else None
        ),
        "transform_model_chosen": str(transform_raw_to_canonical_type),
        "score_sim": float(transform_score_sim) if np.isfinite(transform_score_sim) else None,
        "score_affine": float(transform_score_affine) if np.isfinite(transform_score_affine) else None,
        "score_h": float(transform_score_h) if np.isfinite(transform_score_h) else None,
        "inliers_count_sim": int(transform_inliers_sim),
        "inliers_count_affine": int(transform_inliers_affine),
        "inliers_count_h": int(transform_inliers_h),
        "reproj_median": float(reproj_median_px) if np.isfinite(reproj_median_px) else None,
        "reproj_mean": float(reproj_mean_px) if np.isfinite(reproj_mean_px) else None,
        "reproj_max": float(reproj_max_px) if np.isfinite(reproj_max_px) else None,
        "slice_mode": str(slice_mode),
        "geometry_suspect": bool(geometry_suspect),
        "pitch_final": float(pitch_px),
        "det_cleanup_clusters": det_cleanup_clusters,
        "indexed_points": indexed_points,
        "dedup_meta": dedup_meta,
        "det_fallback_crop_radius_px": float(
            np.clip(
                POSTPROCESS_RECOVERY_CFG["fallback_crop_radius_ratio"] * pitch_px,
                POSTPROCESS_RECOVERY_CFG["fallback_crop_radius_min_px"],
                POSTPROCESS_RECOVERY_CFG["fallback_crop_radius_max_px"],
            )
        ),
    }
    return geometry_detections, qc, debug_payload, canonical_context


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


def _estimate_pitch_px(
    source_detections: List[ChamberDetection],
    roi_bbox: Optional[Any],
    geometry_engine: CrossGeometryEngine
) -> Tuple[float, str]:
    """
    基于检测点估计 pitch（鲁棒 kNN 版本），用于自适应阈值。
    """
    pts = (
        np.array([d.center for d in source_detections], dtype=np.float32)
        if source_detections
        else np.empty((0, 2), dtype=np.float32)
    )
    robust = _estimate_pitch_robust_from_points(pts)
    pitch = float(robust.get("pitch_filtered", float("nan")))
    if not np.isfinite(pitch):
        pitch = float(robust.get("pitch_raw", float("nan")))
    if np.isfinite(pitch):
        pitch_clamped = float(np.clip(
            pitch,
            float(POSTPROCESS_RECOVERY_CFG["pitch_min_px"]),
            float(POSTPROCESS_RECOVERY_CFG["pitch_max_px"]),
        ))
        return pitch_clamped, "knn3_filtered"

    fallback, method = _estimate_pitch_from_roi_or_fallback(
        roi_bbox=roi_bbox,
        geometry_engine=geometry_engine,
    )
    fallback = float(np.clip(
        fallback,
        float(POSTPROCESS_RECOVERY_CFG["pitch_min_px"]),
        float(POSTPROCESS_RECOVERY_CFG["pitch_max_px"]),
    ))
    return fallback, method


def _match_template_to_detections_hungarian(
    fitted_centers: np.ndarray,
    source_detections: List[ChamberDetection],
    match_thresh_px: float
) -> Tuple[Dict[int, int], Dict[int, float], str]:
    """
    template(12点) 与 detections(N点) 的一对一匹配。
    返回:
    - template_idx -> det_idx
    - template_idx -> matched_dist_px
    - assignment_method
    """
    if fitted_centers.ndim != 2 or fitted_centers.shape[1] != 2:
        return {}, {}, "invalid_fitted"
    if not source_detections:
        return {}, {}, "empty_detections"

    det_pts = np.array([d.center for d in source_detections], dtype=np.float32)
    dmat = np.linalg.norm(
        fitted_centers[:, None, :] - det_pts[None, :, :],
        axis=2
    )
    big = 1e6
    # 仅允许严格小于阈值的配对，满足最小契约
    cost = np.where(dmat < float(match_thresh_px), dmat, big)

    mapping: Dict[int, int] = {}
    match_dists: Dict[int, float] = {}

    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        method = "hungarian"
        for r, c in zip(rows.tolist(), cols.tolist()):
            dist = float(dmat[r, c])
            if cost[r, c] >= big or dist >= float(match_thresh_px):
                continue
            mapping[int(r)] = int(c)
            match_dists[int(r)] = dist
        return mapping, match_dists, method

    # scipy 不可用时退化为贪心一对一
    pairs: List[Tuple[float, int, int]] = []
    for r in range(dmat.shape[0]):
        for c in range(dmat.shape[1]):
            dist = float(dmat[r, c])
            if dist < float(match_thresh_px):
                pairs.append((dist, r, c))
    pairs.sort(key=lambda x: x[0])

    used_r = set()
    used_c = set()
    for dist, r, c in pairs:
        if r in used_r or c in used_c:
            continue
        mapping[int(r)] = int(c)
        match_dists[int(r)] = float(dist)
        used_r.add(r)
        used_c.add(c)
    return mapping, match_dists, "greedy_fallback"


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


def _ensure_debug_overlay_from_topology_png(
    run_dir: Path,
    topology_png_path: Optional[Path]
) -> None:
    """
    对失败样本兜底生成 debug_overlay.png（无几何对齐产物时）。
    """
    if topology_png_path is None:
        return
    dst = Path(run_dir) / "debug_overlay.png"
    if dst.exists():
        return
    try:
        shutil.copyfile(str(topology_png_path), str(dst))
    except Exception:
        pass


def _write_blank_features_debug(run_dir: Path, payload: Dict[str, Any]) -> None:
    """
    输出 BLANK 判定专用调试 JSON，便于单独排查 reference arm / blank 误判。
    """
    try:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        blank = payload.get("blank", {}) if isinstance(payload, dict) else {}
        fit = payload.get("fit", {}) if isinstance(payload, dict) else {}
        qc = payload.get("qc", {}) if isinstance(payload, dict) else {}
        out = {
            "chip_id": payload.get("chip_id"),
            "attempt_id": payload.get("attempt_id"),
            "status": payload.get("status"),
            "final_status": payload.get("final_status"),
            "failure_reason": payload.get("failure_reason") or payload.get("final_reason"),
            "raw_image_path": payload.get("raw_image_path"),
            "blank_mode": blank.get("blank_mode"),
            "blank_id_pred": blank.get("blank_id_pred", blank.get("blank_id")),
            "blank_id_old": blank.get("blank_id_old"),
            "blank_id_color": blank.get("blank_id_color"),
            "blank_id_chromaticity": blank.get("blank_id_chromaticity"),
            "blank_status_pred": blank.get("blank_status_pred", blank.get("blank_status")),
            "blank_status": blank.get("blank_status"),
            "blank_confidence": blank.get("blank_confidence"),
            "blank_score": blank.get("blank_score"),
            "blank_score_spread": blank.get("blank_score_spread"),
            "blank_unresolved": blank.get("blank_unresolved"),
            "blank_is_outermost": blank.get("blank_is_outermost"),
            "blank_valid": blank.get("blank_valid"),
            "blank_selected": blank.get("blank_selected"),
            "blank_selected_index": blank.get("blank_selected_index"),
            "reference_arm_pred": blank.get("reference_arm_pred", blank.get("reference_arm")),
            "reference_arm": blank.get("reference_arm"),
            "arm_margin": blank.get("arm_margin"),
            "blank_margin": blank.get("blank_margin"),
            "arm_top2": blank.get("arm_top2"),
            "blank_top2": blank.get("blank_top2"),
            "arm_scores": blank.get("arm_scores"),
            "reference_arm_chamber_scores": blank.get("reference_arm_chamber_scores"),
            "candidate_indices": blank.get("candidate_indices"),
            "blank_scores": blank.get("blank_scores"),
            "score_method": blank.get("score_method"),
            "score_weights": blank.get("score_weights"),
            "chamber_scores_by_template": blank.get("chamber_scores_by_template"),
            "chamber_features_by_template": blank.get("chamber_color_features_by_template"),
            "blank_features": blank.get("blank_features"),
            "pitch_px": payload.get("pitch_px", fit.get("pitch_px")),
            "match_thresh_px": payload.get("match_thresh_px", fit.get("match_thresh_px")),
            "inlier_thresh_px": payload.get("inlier_thresh_px", fit.get("inlier_thresh_px")),
            "fit_success": fit.get("fit_success"),
            "transform_type": fit.get("transform_type"),
            "n_inliers": fit.get("n_inliers"),
            "rmse_px": fit.get("rmse_px"),
            "qc_fail_reasons": qc.get("qc_fail_reasons"),
        }
        with open(run_dir / "debug_blank_features.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception:
        # debug 写盘失败不应影响主流程
        return


def _build_fitted_point_debug(
    fitted_centers: np.ndarray,
    source_detections: List[ChamberDetection],
    visibility: Optional[np.ndarray],
    match_map: Dict[int, int],
    match_dists: Dict[int, float],
    arm_name_to_indices: Dict[str, Tuple[int, int, int]],
    template_ids: Optional[List[str]] = None,
    final_points: Optional[np.ndarray] = None,
    point_sources: Optional[List[str]] = None,
    det_fit_distances: Optional[List[Optional[float]]] = None,
) -> Dict[str, Any]:
    src_centers = (
        np.array([d.center for d in source_detections], dtype=np.float32)
        if source_detections
        else np.empty((0, 2), dtype=np.float32)
    )

    # 防御性去重：即使上游 map 异常，也强制输出为“det_idx 全局唯一”的一对一匹配
    sanitized_pairs: List[Tuple[float, int, int]] = []
    for fit_idx, det_idx in (match_map or {}).items():
        try:
            fi = int(fit_idx)
            dj = int(det_idx)
        except Exception:
            continue
        if fi < 0 or fi >= int(fitted_centers.shape[0]):
            continue
        if dj < 0 or dj >= int(src_centers.shape[0]):
            continue
        dist = match_dists.get(fi)
        if dist is None:
            dist = float(np.linalg.norm(fitted_centers[fi] - src_centers[dj]))
        sanitized_pairs.append((float(dist), fi, dj))

    sanitized_pairs.sort(key=lambda x: (x[0], x[1], x[2]))
    used_fit = set()
    used_det = set()
    unique_match_map: Dict[int, int] = {}
    unique_match_dists: Dict[int, float] = {}
    for dist, fi, dj in sanitized_pairs:
        if fi in used_fit or dj in used_det:
            continue
        unique_match_map[fi] = dj
        unique_match_dists[fi] = dist
        used_fit.add(fi)
        used_det.add(dj)

    n_points = int(fitted_centers.shape[0])
    template_ids = template_ids or [str(i) for i in range(n_points)]
    final_points_arr = np.asarray(final_points, dtype=np.float32) if isinstance(final_points, np.ndarray) else fitted_centers
    if final_points_arr.shape != fitted_centers.shape:
        final_points_arr = fitted_centers
    point_sources = list(point_sources) if isinstance(point_sources, list) else []
    det_fit_distances = list(det_fit_distances) if isinstance(det_fit_distances, list) else []

    fitted_points: List[Dict[str, Any]] = []
    pure_filled_count = 0
    matched_count = 0
    det_used_count = 0
    reject_det_count = 0
    fill_count = 0

    for idx in range(n_points):
        p_fit = (float(fitted_centers[idx, 0]), float(fitted_centers[idx, 1]))
        p_final = (float(final_points_arr[idx, 0]), float(final_points_arr[idx, 1]))
        template_id = str(template_ids[idx]) if idx < len(template_ids) else str(idx)

        if src_centers.shape[0] > 0:
            nearest_det_dist: Optional[float] = float(
                np.min(
                    np.linalg.norm(
                        src_centers - np.array([p_fit[0], p_fit[1]], dtype=np.float32),
                        axis=1
                    )
                )
            )
        else:
            nearest_det_dist = None

        detected_by_model = idx in unique_match_map
        if detected_by_model:
            matched_det_idx = int(unique_match_map[idx])
            match_dist = float(unique_match_dists.get(idx, 0.0))
            matched_count += 1
            pure_filled = False
            p_det = (
                float(src_centers[matched_det_idx, 0]),
                float(src_centers[matched_det_idx, 1]),
            ) if 0 <= matched_det_idx < int(src_centers.shape[0]) else None
        else:
            matched_det_idx = -1
            # 未匹配点不允许“挂名匹配”，match_dist_px 必须为 null
            match_dist = None
            pure_filled = True
            pure_filled_count += 1
            p_det = None

        source = point_sources[idx] if idx < len(point_sources) else ("det" if detected_by_model else "filled")
        if source in ("det", "det_mid"):
            det_used_count += 1
        elif source == "filled" and detected_by_model:
            reject_det_count += 1
        elif source == "filled":
            fill_count += 1

        det_fit_distance = (
            det_fit_distances[idx]
            if idx < len(det_fit_distances)
            else match_dist
        )
        fitted_points.append(
            {
                "template_index": int(idx),
                "template_id": template_id,
                "x": p_final[0],
                "y": p_final[1],
                "visibility": bool(visibility[idx]) if visibility is not None and idx < len(visibility) else None,
                "detected_by_model": bool(detected_by_model),
                "matched_det_idx": matched_det_idx,
                "match_dist_px": match_dist,
                "nearest_det_dist": nearest_det_dist,
                "pure_filled": pure_filled,
                "source": source,
                "p_det": [float(p_det[0]), float(p_det[1])] if p_det is not None else None,
                "p_fit": [p_fit[0], p_fit[1]],
                "p_final": [p_final[0], p_final[1]],
                "det_fit_distance_px": None if det_fit_distance is None else float(det_fit_distance),
            }
        )

    ratio = float(pure_filled_count / max(1, len(fitted_points)))

    matched_count_per_arm: Dict[str, int] = {}
    matched_arm_count = 0
    for arm_name, arm_indices in arm_name_to_indices.items():
        count = int(sum(1 for idx in arm_indices if idx in unique_match_map))
        matched_count_per_arm[arm_name] = count
        if count > 0:
            matched_arm_count += 1
    matched_coverage_arms = float(matched_arm_count / max(1, len(arm_name_to_indices)))

    return {
        "fitted_points": fitted_points,
        "pure_filled_count": int(pure_filled_count),
        "pure_filled_ratio": ratio,
        "matched_count": int(matched_count),
        "det_used_count": int(det_used_count),
        "reject_det_count": int(reject_det_count),
        "fill_count": int(fill_count),
        "matched_count_per_arm": matched_count_per_arm,
        "matched_coverage_arms": matched_coverage_arms,
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
    pitch_px: Optional[float] = None,
    pitch_estimation_method: Optional[str] = None,
    match_thresh_px: Optional[float] = None,
    inlier_thresh_px: Optional[float] = None,
    assignment_method: Optional[str] = None,
    n_inliers: int = 0,
    rmse_px: Optional[float] = None,
    scale: Optional[float] = None,
    rotation_deg: Optional[float] = None,
    coverage_arms: Optional[float] = None,
    inlier_indices: Optional[List[int]] = None,
    fitted_centers: Optional[np.ndarray] = None,
    final_points: Optional[np.ndarray] = None,
    point_sources: Optional[List[str]] = None,
    det_fit_distances: Optional[List[Optional[float]]] = None,
    visibility: Optional[np.ndarray] = None,
    match_map: Optional[Dict[int, int]] = None,
    match_dists: Optional[Dict[int, float]] = None,
    blank_scores: Optional[Dict[int, float]] = None,
    blank_indices: Optional[List[int]] = None,
    blank_confidence: Optional[float] = None,
    blank_status: Optional[str] = None,
    blank_features: Optional[Dict[int, Dict[str, float]]] = None,
    blank_score_spread: Optional[float] = None,
    blank_candidate_indices: Optional[List[int]] = None,
    blank_mode: Optional[str] = None,
    blank_id_old: Optional[int] = None,
    blank_id_color: Optional[int] = None,
    blank_id_chromaticity: Optional[int] = None,
    blank_status_color: Optional[str] = None,
    blank_status_chromaticity: Optional[str] = None,
    reference_arm: Optional[str] = None,
    reference_arm_pred: Optional[str] = None,
    arm_scores: Optional[Dict[str, float]] = None,
    reference_arm_chamber_scores: Optional[Dict[str, float]] = None,
    arm_margin: Optional[float] = None,
    blank_margin: Optional[float] = None,
    chamber_color_features_by_template: Optional[Dict[str, Dict[str, float]]] = None,
    chamber_scores_by_template: Optional[Dict[str, float]] = None,
    chamber_features_by_template: Optional[Dict[str, Dict[str, float]]] = None,
    arm_top2: Optional[List[Dict[str, Any]]] = None,
    blank_top2: Optional[List[Dict[str, Any]]] = None,
    score_method: Optional[str] = None,
    score_weights: Optional[Dict[str, float]] = None,
    template_ids: Optional[List[str]] = None,
    semantic_roles_by_template: Optional[Dict[str, str]] = None,
    semantic_arm_role_by_template: Optional[Dict[str, str]] = None,
    det_priority_stats: Optional[Dict[str, Any]] = None,
    roi_bbox: Optional[Any] = None,
    failure_reason: Optional[str] = None,
    detection_params: Optional[Dict[str, Any]] = None,
    qc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def _as_float(value: Any) -> Optional[float]:
        try:
            f = float(value)
        except Exception:
            return None
        return float(f) if np.isfinite(f) else None

    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except Exception:
            return None

    detections_payload = _serialize_detections_for_debug(source_detections)

    fit_block: Dict[str, Any] = {
        "fit_success": bool(fit_success),
        "transform_type": str(transform_type),
        "n_inliers": int(n_inliers),
        "pitch_px": float(pitch_px) if pitch_px is not None else None,
        "pitch_estimation_method": pitch_estimation_method,
        "match_thresh_px": float(match_thresh_px) if match_thresh_px is not None else None,
        "inlier_thresh_px": float(inlier_thresh_px) if inlier_thresh_px is not None else None,
        "assignment_method": assignment_method,
        "rmse_px": float(rmse_px) if rmse_px is not None and np.isfinite(rmse_px) else None,
        "scale": float(scale) if scale is not None and np.isfinite(scale) else None,
        "rotation_deg": float(rotation_deg) if rotation_deg is not None and np.isfinite(rotation_deg) else None,
        "coverage_arms": float(coverage_arms) if coverage_arms is not None and np.isfinite(coverage_arms) else None,
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
            match_map=match_map or {},
            match_dists=match_dists or {},
            arm_name_to_indices=ARM_NAME_TO_INDICES,
            template_ids=template_ids,
            final_points=final_points,
            point_sources=point_sources,
            det_fit_distances=det_fit_distances,
        )

    blank_scores_dict = {str(int(k)): float(v) for k, v in (blank_scores or {}).items()}
    blank_indices = [int(i) for i in (blank_indices or [])]
    blank_id = int(blank_indices[0]) if blank_indices else None
    arm_scores_payload: Dict[str, float] = {}
    for k, v in (arm_scores or {}).items():
        try:
            f = float(v)
        except Exception:
            continue
        if np.isfinite(f):
            arm_scores_payload[str(k)] = f

    ref_arm_scores_payload: Dict[str, float] = {}
    for k, v in (reference_arm_chamber_scores or {}).items():
        try:
            f = float(v)
        except Exception:
            continue
        if np.isfinite(f):
            ref_arm_scores_payload[str(k)] = f

    chamber_scores_payload: Dict[str, float] = {}
    for tid, score in (chamber_scores_by_template or {}).items():
        try:
            fs = float(score)
        except Exception:
            continue
        if np.isfinite(fs):
            chamber_scores_payload[str(tid)] = fs

    chamber_features_source = chamber_features_by_template
    if not isinstance(chamber_features_source, dict) or not chamber_features_source:
        chamber_features_source = chamber_color_features_by_template or {}
    chamber_color_features_payload: Dict[str, Dict[str, float]] = {}
    for tid, feat in chamber_features_source.items():
        if not isinstance(feat, dict):
            continue
        clean_feat: Dict[str, float] = {}
        for name, value in feat.items():
            try:
                fv = float(value)
            except Exception:
                continue
            if np.isfinite(fv):
                clean_feat[str(name)] = fv
        chamber_color_features_payload[str(tid)] = clean_feat

    blank_mode_norm = str(blank_mode or "").strip().lower()
    blank_id_pred = blank_id
    if blank_mode_norm == "color" and blank_id_color is not None:
        blank_id_pred = int(blank_id_color)
    if blank_mode_norm in {"chromaticity", "color_v2"} and blank_id_chromaticity is not None:
        blank_id_pred = int(blank_id_chromaticity)
    blank_status_pred = str(blank_status or "unresolved")
    blank_unresolved = bool(blank_status_pred == "unresolved")
    blank_is_outermost = bool(blank_id_pred in OUTERMOST_TEMPLATE_INDICES) if blank_id_pred is not None else False
    blank_score = None
    if blank_id_pred is not None:
        blank_score = blank_scores_dict.get(str(int(blank_id_pred)))

    arm_top2_payload: List[Dict[str, Any]] = []
    for item in (arm_top2 or []):
        if not isinstance(item, dict):
            continue
        arm_name = str(item.get("arm", ""))
        score = _as_float(item.get("score"))
        if arm_name and score is not None:
            arm_top2_payload.append({"arm": arm_name, "score": float(score)})

    blank_top2_payload: List[Dict[str, Any]] = []
    for item in (blank_top2 or []):
        if not isinstance(item, dict):
            continue
        tidx = _as_int(item.get("template_index"))
        tid = item.get("template_id")
        score = _as_float(item.get("score"))
        if score is None:
            continue
        blank_top2_payload.append(
            {
                "template_index": tidx,
                "template_id": str(tid) if tid is not None else None,
                "score": float(score),
            }
        )

    score_weights_payload: Dict[str, float] = {}
    for k, v in (score_weights or {}).items():
        fv = _as_float(v)
        if fv is not None:
            score_weights_payload[str(k)] = float(fv)

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
        "pitch_px": float(pitch_px) if pitch_px is not None else None,
        "pitch_estimation_method": pitch_estimation_method,
        "match_thresh_px": float(match_thresh_px) if match_thresh_px is not None else None,
        "inlier_thresh_px": float(inlier_thresh_px) if inlier_thresh_px is not None else None,
        "assignment_method": assignment_method,
        "input": {
            "min_topology_detections": int(min_topology_detections),
            "detection_params": detection_params or {},
        },
        "n_det": int(len(source_detections)),
        "n_det_raw": (
            int(qc.get("n_det_raw"))
            if isinstance(qc, dict) and qc.get("n_det_raw") is not None
            else int(len(source_detections))
        ),
        "n_det_dedup": (
            int(qc.get("n_det_dedup"))
            if isinstance(qc, dict) and qc.get("n_det_dedup") is not None
            else int(len(source_detections))
        ),
        "clusters": (
            list(qc.get("clusters", []))
            if isinstance(qc, dict)
            else []
        ),
        "pitch_final": (
            float(qc.get("pitch_final"))
            if isinstance(qc, dict) and qc.get("pitch_final") is not None
            else (float(pitch_px) if pitch_px is not None else None)
        ),
        "pitch_guard_triggered": (
            bool(qc.get("pitch_guard_triggered"))
            if isinstance(qc, dict)
            else False
        ),
        "fill_ratio": (
            float(qc.get("fill_ratio"))
            if isinstance(qc, dict) and qc.get("fill_ratio") is not None
            else None
        ),
        "used_real_points": (
            int(qc.get("used_real_points"))
            if isinstance(qc, dict) and qc.get("used_real_points") is not None
            else None
        ),
        "model_chosen": (
            str(qc.get("model_chosen"))
            if isinstance(qc, dict) and qc.get("model_chosen") is not None
            else str(transform_type)
        ),
        "reproj_median": (
            float(qc.get("reproj_median"))
            if isinstance(qc, dict) and qc.get("reproj_median") is not None
            else None
        ),
        "reproj_mean": (
            float(qc.get("reproj_mean"))
            if isinstance(qc, dict) and qc.get("reproj_mean") is not None
            else None
        ),
        "geometry_suspect": (
            bool(qc.get("geometry_suspect"))
            if isinstance(qc, dict)
            else False
        ),
        "slice_mode": (
            str(qc.get("slice_mode"))
            if isinstance(qc, dict) and qc.get("slice_mode") is not None
            else "canonical"
        ),
        "indexed_points": (
            list(qc.get("indexed_points", []))
            if isinstance(qc, dict)
            else []
        ),
        "detections": detections_payload,
        "inlier_det_indices": [int(i) for i in (inlier_indices or [])],
        "fit": fit_block,
        "template": template_block,
        "template_ids": template_ids or [str(i) for i in range(12)],
        "final_points_raw": (
            np.asarray(final_points, dtype=np.float32).tolist()
            if isinstance(final_points, np.ndarray)
            else None
        ),
        "point_source": list(point_sources) if isinstance(point_sources, list) else [],
        "det_fit_distance_px": list(det_fit_distances) if isinstance(det_fit_distances, list) else [],
        "fitted_points": fitted_block["fitted_points"],
        "pure_filled_count": int(fitted_block["pure_filled_count"]),
        "pure_filled_ratio": float(fitted_block["pure_filled_ratio"]),
        "matched_count": int(fitted_block["matched_count"]),
        "det_used_count": int(fitted_block.get("det_used_count", 0)),
        "reject_det_count": int(fitted_block.get("reject_det_count", 0)),
        "fill_count": int(fitted_block.get("fill_count", 0)),
        "matched_count_per_arm": fitted_block["matched_count_per_arm"],
        "matched_coverage_arms": float(fitted_block["matched_coverage_arms"]),
        "det_priority": det_priority_stats or {},
        "blank": {
            "candidate_indices": [int(i) for i in (blank_candidate_indices or list(OUTERMOST_TEMPLATE_INDICES))],
            "blank_scores": blank_scores_dict,
            "blank_id": blank_id,
            "blank_id_old": None if blank_id_old is None else int(blank_id_old),
            "blank_id_color": None if blank_id_color is None else int(blank_id_color),
            "blank_id_chromaticity": None if blank_id_chromaticity is None else int(blank_id_chromaticity),
            "blank_id_pred": None if blank_id_pred is None else int(blank_id_pred),
            "blank_mode": blank_mode,
            "blank_confidence": float(blank_confidence) if blank_confidence is not None else None,
            "blank_status": blank_status,
            "blank_status_color": blank_status_color,
            "blank_status_chromaticity": blank_status_chromaticity,
            "blank_status_pred": blank_status_pred,
            "blank_score_spread": float(blank_score_spread) if blank_score_spread is not None else None,
            "blank_score": blank_score,
            "blank_unresolved": blank_unresolved,
            "blank_is_outermost": bool(blank_is_outermost),
            "blank_valid": bool((blank_id_pred is not None) and blank_is_outermost and (blank_status_pred != "unresolved")),
            "blank_selected": (
                str(template_ids[int(blank_id_pred)])
                if isinstance(template_ids, list) and blank_id_pred is not None and 0 <= int(blank_id_pred) < len(template_ids)
                else None
            ),
            "blank_selected_index": None if blank_id_pred is None else int(blank_id_pred),
            "reference_arm": reference_arm,
            "reference_arm_pred": reference_arm_pred or reference_arm,
            "arm_scores": arm_scores_payload,
            "reference_arm_chamber_scores": ref_arm_scores_payload,
            "arm_margin": float(arm_margin) if arm_margin is not None and np.isfinite(arm_margin) else None,
            "blank_margin": float(blank_margin) if blank_margin is not None and np.isfinite(blank_margin) else None,
            "arm_top2": arm_top2_payload,
            "blank_top2": blank_top2_payload,
            "blank_features": {
                str(k): {kk: float(vv) for kk, vv in feat.items()}
                for k, feat in (blank_features or {}).items()
            },
            "chamber_scores_by_template": chamber_scores_payload,
            "chamber_color_features_by_template": chamber_color_features_payload,
            "score_method": score_method
            or (
                "w1*abs(delta_f_rg) + w2*delta_f_red + w3*delta_f_s"
                if str(blank_mode or "").lower() == "color"
                else "0.45*V_low + 0.25*|dV| + 0.20*S_med + 0.40*sat_ratio*255"
            ),
            "score_weights": score_weights_payload,
        },
        "semantic_roles_by_template": semantic_roles_by_template or {},
        "semantic_arm_role_by_template": semantic_arm_role_by_template or {},
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
    force_blank_if_missing: bool,
    match_map: Optional[Dict[int, int]] = None
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
    effective_match_map = (
        dict(match_map)
        if match_map is not None
        else _build_unique_assignment(fitted, source_centers, match_radius)
    )

    for fit_idx, (cx, cy) in enumerate(fitted):
        class_id = int(class_id_lit)
        confidence = 0.0

        src_idx = effective_match_map.get(fit_idx)
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
    save_debug: bool = True,
    export_geo_even_if_blank_unresolved: Optional[bool] = None,
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
    selected_canonical_context: Optional[Dict[str, Any]] = None
    selected_qc: Dict[str, Any] = {}
    selected_source = "json"

    # 始终保存原图，便于失败样本审计
    cv2.imwrite(str(run_dir / "raw.png"), raw_image)

    json_detection_params: Dict[str, Any] = {
        "source_json": str(detections_json_path),
        "adaptive_config_from_json": payload.get("adaptive_config"),
        "preprocess_mode": "from_json",
    }
    json_roi_bbox = payload.get("roi_bbox")

    # ---------- Try 1: detections from JSON ----------
    try:
        geometry_detections, qc, attempt_payload, canonical_context = _run_topology_refine_postprocess(
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
            geometry_engine=effective_geometry_engine,
            export_geo_even_if_blank_unresolved=export_geo_even_if_blank_unresolved,
        )
        debug_dump_topology(run_dir, raw_image, attempt_payload, suffix="_try1")
        attempts.append(attempt_payload)
        selected_attempt_payload = attempt_payload
        selected_geometry_detections = geometry_detections
        selected_canonical_context = canonical_context
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

        first_bucket = _classify_postprocess_failure_bucket(str(first_err))
        allow_fallback_for_bucket = first_bucket in {"fail_n_det_lt_min", "fail_topology_fit"}
        if (not enable_fallback_detection) or (not allow_fallback_for_bucket):
            final_payload = dict(first_payload)
            final_payload["used_fallback"] = False
            final_payload["final_status"] = "failed"
            final_payload["final_reason"] = str(first_err)
            final_payload["fail_bucket"] = first_bucket
            final_payload["attempts"] = attempts
            final_payload["selected_attempt_id"] = "try1_json"
            final_paths = debug_dump_topology(run_dir, raw_image, final_payload, suffix=None)
            _write_blank_features_debug(run_dir, final_payload)
            _ensure_debug_overlay_from_topology_png(run_dir, final_paths.get("png"))
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
            geometry_detections, qc, second_payload, canonical_context = _run_topology_refine_postprocess(
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
                geometry_engine=effective_geometry_engine,
                export_geo_even_if_blank_unresolved=export_geo_even_if_blank_unresolved,
            )
            second_payload["fallback_reason"] = str(first_err)
            debug_dump_topology(run_dir, raw_image, second_payload, suffix="_try2")
            attempts.append(second_payload)

            selected_attempt_payload = second_payload
            selected_geometry_detections = geometry_detections
            selected_canonical_context = canonical_context
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
            final_payload["fail_bucket"] = _classify_postprocess_failure_bucket(str(second_err))
            final_payload["attempts"] = attempts
            final_payload["selected_attempt_id"] = "try2_fallback"
            final_paths = debug_dump_topology(run_dir, raw_image, final_payload, suffix=None)
            _write_blank_features_debug(run_dir, final_payload)
            _ensure_debug_overlay_from_topology_png(run_dir, final_paths.get("png"))
            raise RuntimeError(str(second_err)) from second_err

    if selected_geometry_detections is None or selected_attempt_payload is None or selected_canonical_context is None:
        raise RuntimeError(f"[{final_chip_id}] unexpected empty postprocess output")

    selected_qc.update(
        {
            "source_detections_json": str(detections_json_path),
            "input_detection_count": int(len(detections_from_json)),
            "selected_detection_count": int(len(selected_geometry_detections)),
            "min_topology_detections": int(min_required),
            "used_fallback": bool(selected_source != "json"),
        }
    )
    semantic_ready = bool(selected_qc.get("semantic_ready", False))
    selected_qc["semantic_ready"] = semantic_ready
    selected_qc["blank_pass"] = semantic_ready
    selected_qc["geo_success"] = False
    selected_qc["usable_for_stage2"] = False
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
    _write_blank_features_debug(run_dir, final_debug_payload)

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
        retry_attempt=1 if selected_source != "json" else 0,
        canonical_context=selected_canonical_context
    )

    # 几何后验门控：若 canonical 对齐重投影异常，则判失败
    geom_debug = (
        effective_geometry_engine.last_process_debug
        if isinstance(getattr(effective_geometry_engine, "last_process_debug", None), dict)
        else {}
    )
    reproj_mean = geom_debug.get("reprojection_error_mean_px")
    reproj_max = geom_debug.get("reprojection_error_max_px")
    center_offset_max = geom_debug.get("slice_center_offset_max_px")
    slice_mode = str(selected_qc.get("slice_mode", selected_canonical_context.get("slice_mode", "canonical")))
    geometry_suspect = bool(selected_qc.get("geometry_suspect", False))
    soft_reproj_limit = float(POST_QC_MAX_GEOMETRY_REPROJ_RATIO * float(selected_qc.get("pitch_px", 50.0)))
    hard_reproj_limit = float(POST_QC_HARD_GEOMETRY_REPROJ_RATIO * float(selected_qc.get("pitch_px", 50.0)))
    soft_offset_limit = float(POST_QC_MAX_SLICE_CENTER_OFFSET_PX)
    hard_offset_limit = float(POST_QC_HARD_SLICE_CENTER_OFFSET_PX)
    hard_fail_reasons: List[str] = []
    soft_warn_reasons: List[str] = []
    if slice_mode == "det_fallback":
        soft_warn_reasons.append("slice_mode=det_fallback")
        if geometry_suspect:
            soft_warn_reasons.append("geometry_suspect=true")
        if reproj_mean is not None and np.isfinite(float(reproj_mean)):
            reproj_val = float(reproj_mean)
            if reproj_val > soft_reproj_limit:
                soft_warn_reasons.append(f"geometry_reprojection_mean>{soft_reproj_limit:.2f}")
        if center_offset_max is not None and np.isfinite(float(center_offset_max)):
            center_offset_val = float(center_offset_max)
            if center_offset_val > soft_offset_limit:
                soft_warn_reasons.append(f"slice_center_offset_max>{soft_offset_limit:.1f}")
    else:
        if reproj_mean is None or not np.isfinite(float(reproj_mean)):
            hard_fail_reasons.append("geometry_reprojection_missing")
        else:
            reproj_val = float(reproj_mean)
            if reproj_val > hard_reproj_limit:
                hard_fail_reasons.append(f"geometry_reprojection_mean>{hard_reproj_limit:.2f}")
            elif reproj_val > soft_reproj_limit:
                soft_warn_reasons.append(f"geometry_reprojection_mean>{soft_reproj_limit:.2f}")
        if center_offset_max is None or not np.isfinite(float(center_offset_max)):
            hard_fail_reasons.append("slice_center_offset_missing")
        else:
            center_offset_val = float(center_offset_max)
            if center_offset_val > hard_offset_limit:
                hard_fail_reasons.append(f"slice_center_offset_max>{hard_offset_limit:.1f}")
            elif center_offset_val > soft_offset_limit:
                soft_warn_reasons.append(f"slice_center_offset_max>{soft_offset_limit:.1f}")
    geo_success = len(hard_fail_reasons) == 0
    geo_quality_level = "bad"
    if geo_success:
        geo_quality_level = "good" if (len(soft_warn_reasons) == 0 and not geometry_suspect and slice_mode == "canonical") else "warn"
    selected_qc["geo_success"] = bool(geo_success)
    selected_qc["geo_quality_level"] = str(geo_quality_level)
    selected_qc["geometry_suspect"] = bool(geometry_suspect)
    selected_qc["slice_mode"] = str(slice_mode)
    selected_qc["geo_hard_fail_reasons"] = list(hard_fail_reasons)
    selected_qc["geo_soft_warn_reasons"] = list(soft_warn_reasons)
    selected_qc["reprojection_error_mean_px"] = (
        None if reproj_mean is None else float(reproj_mean)
    )
    selected_qc["reprojection_error_max_px"] = (
        None if reproj_max is None else float(reproj_max)
    )
    selected_qc["slice_center_offset_max_px"] = (
        None if center_offset_max is None else float(center_offset_max)
    )
    selected_qc["usable_for_stage2"] = bool(geo_success and semantic_ready)
    selected_qc["semantic_ready"] = bool(semantic_ready)
    selected_qc["blank_pass"] = bool(semantic_ready)
    if result.quality_metrics is None:
        result.quality_metrics = {}
    result.quality_metrics["geo_success"] = bool(geo_success)
    result.quality_metrics["semantic_ready"] = bool(semantic_ready)
    result.quality_metrics["blank_pass"] = bool(semantic_ready)
    result.quality_metrics["usable_for_stage2"] = bool(geo_success and semantic_ready)
    result.quality_metrics["geo_quality_level"] = str(geo_quality_level)
    result.quality_metrics["geometry_suspect"] = bool(geometry_suspect)
    result.quality_metrics["slice_mode"] = str(slice_mode)
    result.quality_metrics["reprojection_error_mean_px"] = (
        None if reproj_mean is None else float(reproj_mean)
    )
    result.quality_metrics["reprojection_error_max_px"] = (
        None if reproj_max is None else float(reproj_max)
    )
    result.quality_metrics["slice_center_offset_max_px"] = (
        None if center_offset_max is None else float(center_offset_max)
    )
    result.quality_metrics["geo_hard_fail_reasons"] = list(hard_fail_reasons)
    result.quality_metrics["geo_soft_warn_reasons"] = list(soft_warn_reasons)
    result.quality_gate_passed = bool(geo_success)

    # 保存 geometry debug 工件
    if geom_debug:
        overlay = geom_debug.get("debug_overlay_raw")
        if isinstance(overlay, np.ndarray):
            cv2.imwrite(str(run_dir / "debug_overlay.png"), overlay)
        raw_slices = geom_debug.get("raw_slices")
        if isinstance(raw_slices, np.ndarray):
            np.savez_compressed(run_dir / "raw_slices.npz", slices=raw_slices)
        geometry_payload = {
            "chip_id": final_chip_id,
            "status": "failed" if hard_fail_reasons else ("warn" if soft_warn_reasons else "success"),
            "reprojection_error_mean_px": reproj_mean,
            "reprojection_error_max_px": reproj_max,
            "reprojection_error_threshold_px": soft_reproj_limit,
            "reprojection_error_hard_threshold_px": hard_reproj_limit,
            "slice_center_offset_max_px": center_offset_max,
            "slice_center_offset_threshold_px": soft_offset_limit,
            "slice_center_offset_hard_threshold_px": hard_offset_limit,
            "template_ids": geom_debug.get("template_ids"),
            "semantic_order_clockwise_from_blank": geom_debug.get("semantic_order_clockwise_from_blank"),
            "semantic_roles_by_template": geom_debug.get("semantic_roles_by_template"),
            "semantic_arm_role_by_template": geom_debug.get("semantic_arm_role_by_template"),
            "blank_idx": geom_debug.get("blank_idx"),
            "blank_status": selected_qc.get("blank_status"),
            "reference_arm_pred": selected_qc.get("reference_arm_pred"),
            "blank_id_pred": selected_qc.get("blank_id_pred"),
            "geo_success": bool(geo_success),
            "geo_quality_level": str(geo_quality_level),
            "geometry_suspect": bool(geometry_suspect),
            "slice_mode": str(slice_mode),
            "semantic_ready": bool(semantic_ready),
            "transform_type": geom_debug.get("transform_type"),
            "transform_matrix_raw_to_canonical": geom_debug.get("transform_matrix_raw_to_canonical"),
            "inverse_transform_matrix_canonical_to_raw": geom_debug.get("inverse_transform_matrix_canonical_to_raw"),
            "fitted_points_raw": geom_debug.get("fitted_points_raw"),
            "topology_fitted_points_raw": geom_debug.get("topology_fitted_points_raw"),
            "point_source": geom_debug.get("point_source"),
            "det_fit_distance_px": geom_debug.get("det_fit_distance_px"),
            "canonical_points": geom_debug.get("canonical_points"),
            "projected_raw_points": geom_debug.get("projected_raw_points"),
            "fill_ratio": selected_qc.get("fill_ratio"),
            "used_real_points": selected_qc.get("used_real_points"),
            "model_chosen": selected_qc.get("model_chosen"),
            "score_sim": selected_qc.get("score_sim"),
            "score_affine": selected_qc.get("score_affine"),
            "score_h": selected_qc.get("score_h"),
            "hard_fail_reasons": hard_fail_reasons,
            "soft_warn_reasons": soft_warn_reasons,
        }
        with open(run_dir / "debug_geometry_alignment.json", "w", encoding="utf-8") as f:
            json.dump(geometry_payload, f, ensure_ascii=False, indent=2)

    if hard_fail_reasons:
        final_debug_payload["final_status"] = "failed"
        final_debug_payload["final_reason"] = "postprocess_qc_failed: geometry_alignment_failed"
        final_debug_payload["fail_bucket"] = "fail_geometry_alignment"
        final_debug_payload["qc"]["geometry_fail_reasons"] = hard_fail_reasons
        final_debug_payload["qc"]["geo_quality_level"] = "bad"
        final_paths = debug_dump_topology(run_dir, raw_image, final_debug_payload, suffix=None)
        _write_blank_features_debug(run_dir, final_debug_payload)
        _ensure_debug_overlay_from_topology_png(run_dir, final_paths.get("png"))
        raise RuntimeError(
            f"postprocess_qc_failed: geometry_alignment_failed ({','.join(hard_fail_reasons)})"
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
    save_debug: bool = True,
    export_geo_even_if_blank_unresolved: Optional[bool] = None,
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
                save_debug=save_debug,
                export_geo_even_if_blank_unresolved=export_geo_even_if_blank_unresolved,
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
