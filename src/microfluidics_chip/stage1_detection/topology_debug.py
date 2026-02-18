"""
Stage1 topology/postprocess debug dump utilities.

统一输出:
- debug_stage1_topology*.png
- debug_stage1_topology*.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

import cv2
import numpy as np


def _as_int_point(x: float, y: float) -> Tuple[int, int]:
    return int(round(float(x))), int(round(float(y)))


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if np.isfinite(f):
        return f
    return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _normalize_bbox(bbox: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x1 = _safe_int(bbox[0])
    y1 = _safe_int(bbox[1])
    x2 = _safe_int(bbox[2])
    y2 = _safe_int(bbox[3])
    if None in (x1, y1, x2, y2):
        return None
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _put_legend(vis: np.ndarray, lines: Sequence[str]) -> None:
    x0, y0 = 10, 20
    line_h = 18
    max_len = 0
    for ln in lines:
        (w, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        max_len = max(max_len, w)
    h = line_h * len(lines) + 10
    cv2.rectangle(vis, (x0 - 6, y0 - 16), (x0 + max_len + 12, y0 + h), (0, 0, 0), -1)
    cv2.rectangle(vis, (x0 - 6, y0 - 16), (x0 + max_len + 12, y0 + h), (80, 80, 80), 1)
    for i, ln in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(
            vis, ln, (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA
        )


def _render_debug_image(raw_image: np.ndarray, payload: Dict[str, Any]) -> np.ndarray:
    vis = raw_image.copy()
    h, w = vis.shape[:2]

    detections = payload.get("detections", []) or []
    inlier_det_indices = set(int(i) for i in (payload.get("inlier_det_indices", []) or []))
    fitted_points = payload.get("fitted_points", []) or []
    blank_info = payload.get("blank", {}) or {}
    blank_id = _safe_int(blank_info.get("blank_id"))
    blank_id_pred = _safe_int(blank_info.get("blank_id_pred"))
    blank_id_old = _safe_int(blank_info.get("blank_id_old"))
    blank_id_color = _safe_int(blank_info.get("blank_id_color"))
    blank_id_chroma = _safe_int(blank_info.get("blank_id_chromaticity"))
    blank_mode = str(blank_info.get("blank_mode", "unknown"))
    blank_status = str(blank_info.get("blank_status", "unknown"))
    blank_status_pred = str(blank_info.get("blank_status_pred", blank_status))
    blank_status_color = str(blank_info.get("blank_status_color", "unknown"))
    blank_status_chroma = str(blank_info.get("blank_status_chromaticity", "unknown"))
    blank_margin = _safe_float(blank_info.get("blank_margin"))
    arm_margin = _safe_float(blank_info.get("arm_margin"))
    reference_arm = blank_info.get("reference_arm")
    reference_arm_pred = blank_info.get("reference_arm_pred", reference_arm)
    arm_scores = blank_info.get("arm_scores", {}) if isinstance(blank_info.get("arm_scores"), dict) else {}
    ref_arm_ch_scores = (
        blank_info.get("reference_arm_chamber_scores", {})
        if isinstance(blank_info.get("reference_arm_chamber_scores"), dict)
        else {}
    )
    candidate_indices = set(int(i) for i in (blank_info.get("candidate_indices", []) or []))

    roi_bbox = _normalize_bbox(payload.get("roi_bbox"), w, h)
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            vis, "ROI", (x1 + 4, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA
        )

    # 红点: all detections
    for idx, det in enumerate(detections):
        cx = _safe_float(det.get("x"))
        cy = _safe_float(det.get("y"))
        if cx is None or cy is None:
            continue
        p = _as_int_point(cx, cy)
        cv2.circle(vis, p, 3, (0, 0, 255), -1)
        # 绿点: inliers
        if idx in inlier_det_indices:
            cv2.circle(vis, p, 7, (0, 255, 0), 1)

    # 蓝点: fitted 12; pure-filled 空心圈
    for pt in fitted_points:
        tid = _safe_int(pt.get("template_index"))
        tid_name = str(pt.get("template_id", tid if tid is not None else ""))
        cx = _safe_float(pt.get("x"))
        cy = _safe_float(pt.get("y"))
        if tid is None or cx is None or cy is None:
            continue
        p = _as_int_point(cx, cy)
        pure_filled = bool(pt.get("pure_filled", False))
        arm_prefix = tid_name[:1].upper() if tid_name else ""
        arm_name = {"U": "Up", "R": "Right", "D": "Down", "L": "Left"}.get(arm_prefix, "")
        is_reference_arm = bool(reference_arm) and arm_name == str(reference_arm)

        if pure_filled:
            cv2.circle(vis, p, 9, (255, 0, 255), 2)  # hollow marker for pure-filled
        if is_reference_arm:
            cv2.circle(vis, p, 11, (0, 165, 255), 2)  # highlight reference arm
        cv2.circle(vis, p, 4, (255, 0, 0), -1)
        cv2.line(vis, (p[0] - 5, p[1]), (p[0] + 5, p[1]), (255, 0, 0), 1)
        cv2.line(vis, (p[0], p[1] - 5), (p[0], p[1] + 5), (255, 0, 0), 1)
        cv2.putText(
            vis, tid_name, (p[0] + 6, p[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1, cv2.LINE_AA
        )

        if tid in candidate_indices:
            cv2.circle(vis, p, 13, (0, 255, 255), 2)
            cv2.putText(
                vis, "C", (p[0] - 4, p[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
            )
        final_blank_idx = blank_id_pred if blank_id_pred is not None else blank_id
        if final_blank_idx is not None and tid == final_blank_idx:
            cv2.circle(vis, p, 18, (0, 0, 255), 2)
            cv2.putText(
                vis, "BLANK", (p[0] - 24, p[1] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA
            )

    fit_info = payload.get("fit", {}) or {}
    n_det_raw = payload.get("n_det_raw", payload.get("n_det"))
    n_det_dedup = payload.get("n_det_dedup", payload.get("n_det"))
    fill_ratio = payload.get("fill_ratio")
    used_real_points = payload.get("used_real_points")
    model_chosen = payload.get("model_chosen", fit_info.get("transform_type", "n/a"))
    reproj_median = payload.get("reproj_median")
    reproj_mean = payload.get("reproj_mean")
    slice_mode = payload.get("slice_mode", "canonical")
    geometry_suspect = payload.get("geometry_suspect", False)
    pitch_final = payload.get("pitch_final", payload.get("pitch_px", fit_info.get("pitch_px")))

    arm_line = "arm_scores=Up:n/a Right:n/a Down:n/a Left:n/a"
    if arm_scores:
        def fmt(name: str) -> str:
            val = _safe_float(arm_scores.get(name))
            return "n/a" if val is None else f"{val:.3f}"
        arm_line = (
            f"arm_scores=Up:{fmt('Up')} Right:{fmt('Right')} "
            f"Down:{fmt('Down')} Left:{fmt('Left')}"
        )
    lines = [
        f"attempt={payload.get('attempt_id', 'n/a')} status={payload.get('status', 'unknown')}",
        f"n_det_raw={n_det_raw} n_det_dedup={n_det_dedup} n_inliers={fit_info.get('n_inliers', 0)} rmse={fit_info.get('rmse_px', 'n/a')}",
        f"fit={fit_info.get('fit_success', False)} transform={fit_info.get('transform_type', 'n/a')}"
        f" scale={fit_info.get('scale', 'n/a')} rot={fit_info.get('rotation_deg', 'n/a')}",
        f"pitch_final={pitch_final} fill_ratio={fill_ratio} used_real={used_real_points}",
        f"model={model_chosen} reproj_median={reproj_median} reproj_mean={reproj_mean}",
        f"slice_mode={slice_mode} geometry_suspect={int(bool(geometry_suspect))}",
        f"blank_mode={blank_mode} blank_id={blank_id_pred if blank_id_pred is not None else blank_id} old={blank_id_old} color={blank_id_color} chroma={blank_id_chroma}",
        f"blank_status={blank_status_pred} raw={blank_status} color={blank_status_color} chroma={blank_status_chroma} arm_margin={arm_margin} blank_margin={blank_margin}",
        arm_line,
        f"reference_arm={reference_arm_pred or 'n/a'} raw={reference_arm or 'n/a'} ref_scores={ref_arm_ch_scores or {}}",
        "red=det green=inlier blue=fitted orange=reference-arm magenta=pure-filled yellow=outermost",
    ]
    if payload.get("failure_reason"):
        lines.append(f"reason={payload['failure_reason']}")
    _put_legend(vis, lines)
    return vis


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def debug_dump_topology(
    output_dir: Path,
    raw_image: np.ndarray,
    payload: Dict[str, Any],
    suffix: Optional[str] = None,
) -> Dict[str, Path]:
    """
    保存 topology debug png/json。

    :param output_dir: 输出目录
    :param raw_image: 原图（坐标系基准）
    :param payload: debug 信息字典（会写入 JSON）
    :param suffix: 可选后缀（例如 "_try1"）
    :return: {"png": Path, "json": Path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name_suffix = suffix or ""
    png_path = output_dir / f"debug_stage1_topology{name_suffix}.png"
    json_path = output_dir / f"debug_stage1_topology{name_suffix}.json"

    vis = _render_debug_image(raw_image, payload)
    cv2.imwrite(str(png_path), vis)

    payload_to_dump = _json_safe(payload)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload_to_dump, f, ensure_ascii=False, indent=2)

    return {"png": png_path, "json": json_path}
