#!/usr/bin/env python3
"""
Batch validation for template-driven Stage1 postprocess.

This script can optionally dump three-way diagnostics in raw-image coordinates:
- P_det: YOLO/raw detections centers
- P_fit: topology fitted/refilled centers (12)
- P_slice_backproj: slicing canonical centers projected back to raw coords

Outputs:
- threeway_summary.csv
- per-sample debug_threeway_overlay.png
- per-sample debug_threeway_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microfluidics_chip.core.config import load_config_from_yaml, TopologyConfig  # noqa: E402
from microfluidics_chip.pipelines.stage1 import run_stage1_postprocess_from_json  # noqa: E402


def _discover_jsons(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for name in ("adaptive_yolo_raw_detections.json", "yolo_raw_detections.json"):
        files.extend(input_dir.rglob(name))
    return sorted(set(files))


def _safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None:
            return default
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _dist(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> float:
    if a is None or b is None:
        return float("nan")
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _mean_max(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.max(vals))


def _quantiles(values: Sequence[float]) -> Tuple[float, float, float]:
    vals = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=np.float32)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(vals, 50)),
        float(np.percentile(vals, 90)),
        float(np.percentile(vals, 95)),
    )


def _read_slice_count(run_dir: Path) -> int:
    npz_path = run_dir / "chamber_slices.npz"
    if not npz_path.exists():
        return 0
    try:
        with np.load(npz_path) as data:
            if "slices" not in data:
                return 0
            return int(data["slices"].shape[0])
    except Exception:
        return 0


def _extract_det_points(detections: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(detections, list):
        return out

    for i, det in enumerate(detections):
        if not isinstance(det, dict):
            continue

        center = det.get("center")
        cx: Optional[float] = None
        cy: Optional[float] = None

        if isinstance(center, (list, tuple)) and len(center) == 2:
            cx = _safe_float(center[0], default=float("nan"))
            cy = _safe_float(center[1], default=float("nan"))

        if (cx is None or cy is None or not np.isfinite(cx) or not np.isfinite(cy)):
            # fallback: parse bbox; default treat as (x, y, w, h)
            bbox = det.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x = _safe_float(bbox[0], default=float("nan"))
                y = _safe_float(bbox[1], default=float("nan"))
                a = _safe_float(bbox[2], default=float("nan"))
                b = _safe_float(bbox[3], default=float("nan"))
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(a) and np.isfinite(b):
                    cx = float(x + a * 0.5)
                    cy = float(y + b * 0.5)

        if cx is None or cy is None or not np.isfinite(cx) or not np.isfinite(cy):
            continue

        out.append(
            {
                "det_idx": int(det.get("det_idx", i)),
                "x": float(cx),
                "y": float(cy),
                "conf": _safe_float(det.get("conf", det.get("confidence", float("nan")))),
                "class_id": _safe_int(det.get("class_id", -1)),
            }
        )
    return out


def _extract_topology_points(
    topo: Dict[str, Any],
    geom: Dict[str, Any],
) -> Tuple[List[str], List[Optional[Tuple[float, float]]], List[int], List[bool], List[bool]]:
    # template ids priority: topo -> geom -> synthetic 12
    template_ids = topo.get("template_ids")
    if not isinstance(template_ids, list) or len(template_ids) == 0:
        template_ids = geom.get("template_ids")
    if not isinstance(template_ids, list) or len(template_ids) == 0:
        template_ids = [str(i) for i in range(12)]

    n = len(template_ids)
    if n < 12:
        n = 12
        template_ids = template_ids + [str(i) for i in range(len(template_ids), 12)]

    fit_pts: List[Optional[Tuple[float, float]]] = [None] * n
    matched_det_idx: List[int] = [-1] * n
    pure_filled: List[bool] = [True] * n
    detected_by_model: List[bool] = [False] * n

    fitted_points = topo.get("fitted_points", [])
    if isinstance(fitted_points, list):
        for fp in fitted_points:
            if not isinstance(fp, dict):
                continue
            idx = _safe_int(fp.get("template_index"), default=-1)
            if idx < 0 or idx >= n:
                continue
            x = _safe_float(fp.get("x"))
            y = _safe_float(fp.get("y"))
            if np.isfinite(x) and np.isfinite(y):
                fit_pts[idx] = (float(x), float(y))
            matched_det_idx[idx] = _safe_int(fp.get("matched_det_idx"), default=-1)
            pure_filled[idx] = bool(fp.get("pure_filled", False))
            detected_by_model[idx] = bool(fp.get("detected_by_model", False))

    # fallback from geometry if topology payload missing/legacy
    if any(p is None for p in fit_pts):
        gfit = geom.get("fitted_points_raw", [])
        if isinstance(gfit, list):
            for i in range(min(n, len(gfit))):
                pt = gfit[i]
                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                    x = _safe_float(pt[0])
                    y = _safe_float(pt[1])
                    if np.isfinite(x) and np.isfinite(y) and fit_pts[i] is None:
                        fit_pts[i] = (float(x), float(y))

    return template_ids[:n], fit_pts[:n], matched_det_idx[:n], pure_filled[:n], detected_by_model[:n]


def _affine_apply(M_2x3: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.hstack([points.astype(np.float32), ones])
    return (M_2x3.astype(np.float32) @ homo.T).T


def _extract_slice_backproj_points(
    geom: Dict[str, Any],
    n_templates: int,
) -> List[Optional[Tuple[float, float]]]:
    out: List[Optional[Tuple[float, float]]] = [None] * n_templates

    proj = geom.get("projected_raw_points")
    if isinstance(proj, list) and len(proj) > 0:
        for i in range(min(n_templates, len(proj))):
            pt = proj[i]
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                x = _safe_float(pt[0])
                y = _safe_float(pt[1])
                if np.isfinite(x) and np.isfinite(y):
                    out[i] = (float(x), float(y))
        return out

    # fallback: inverse(T_raw->canonical) * canonical_points
    invM = geom.get("inverse_transform_matrix_canonical_to_raw")
    canonical_points = geom.get("canonical_points")
    if (
        isinstance(invM, list)
        and len(invM) == 2
        and isinstance(canonical_points, list)
        and len(canonical_points) > 0
    ):
        try:
            M = np.asarray(invM, dtype=np.float32)
            pts = np.asarray(canonical_points, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 2:
                back = _affine_apply(M, pts)
                for i in range(min(n_templates, back.shape[0])):
                    out[i] = (float(back[i, 0]), float(back[i, 1]))
        except Exception:
            pass

    return out


def _resolve_raw_image_path(
    run_dir: Path,
    det_json: Path,
    det_payload: Dict[str, Any],
    topo_payload: Dict[str, Any],
) -> Optional[Path]:
    candidates: List[Path] = []

    candidates.append(run_dir / "raw.png")

    topo_raw = topo_payload.get("raw_image_path")
    if isinstance(topo_raw, str) and topo_raw:
        p = Path(topo_raw)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(det_json.parent / topo_raw)
            candidates.append(run_dir / topo_raw)

    det_raw = det_payload.get("raw_image_path")
    if isinstance(det_raw, str) and det_raw:
        p = Path(det_raw)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(det_json.parent / det_raw)

    candidates.append(det_json.parent / "raw.png")

    for p in candidates:
        if p.exists():
            return p
    return None


def _build_threeway_metrics(
    *,
    template_ids: List[str],
    det_points: List[Dict[str, Any]],
    fit_points_raw: List[Optional[Tuple[float, float]]],
    slice_points_raw: List[Optional[Tuple[float, float]]],
    matched_det_idx: List[int],
    pure_filled: List[bool],
    detected_by_model: List[bool],
    pitch_px: float,
) -> Dict[str, Any]:
    n = len(template_ids)

    det_map: Dict[int, Tuple[float, float]] = {}
    for d in det_points:
        idx = _safe_int(d.get("det_idx"), -1)
        x = _safe_float(d.get("x"))
        y = _safe_float(d.get("y"))
        if idx >= 0 and np.isfinite(x) and np.isfinite(y):
            det_map[idx] = (float(x), float(y))

    per_template: List[Dict[str, Any]] = []
    e1_vals: List[float] = []
    e2_vals: List[float] = []
    e3_vals: List[float] = []

    for i in range(n):
        t_id = str(template_ids[i])
        fit_pt = fit_points_raw[i] if i < len(fit_points_raw) else None
        slc_pt = slice_points_raw[i] if i < len(slice_points_raw) else None
        det_idx = matched_det_idx[i] if i < len(matched_det_idx) else -1
        is_pure = bool(pure_filled[i]) if i < len(pure_filled) else True
        is_detected = bool(detected_by_model[i]) if i < len(detected_by_model) else False

        if is_pure or not is_detected:
            det_idx = -1

        det_pt = det_map.get(det_idx)

        e1 = _dist(det_pt, fit_pt)
        e2 = _dist(fit_pt, slc_pt)
        e3 = _dist(det_pt, slc_pt)

        if np.isfinite(e1):
            e1_vals.append(float(e1))
        if np.isfinite(e2):
            e2_vals.append(float(e2))
        if np.isfinite(e3):
            e3_vals.append(float(e3))

        per_template.append(
            {
                "template_index": int(i),
                "template_id": t_id,
                "matched_det_idx": int(det_idx),
                "is_pure_filled": bool(is_pure),
                "detected_by_model": bool(is_detected),
                "fit_raw": None if fit_pt is None else [float(fit_pt[0]), float(fit_pt[1])],
                "slice_raw_backproj": None if slc_pt is None else [float(slc_pt[0]), float(slc_pt[1])],
                "dist_det_to_fit_px": None if not np.isfinite(e1) else float(e1),
                "dist_fit_to_slice_px": None if not np.isfinite(e2) else float(e2),
                "dist_det_to_slice_px": None if not np.isfinite(e3) else float(e3),
            }
        )

    e1_mean, e1_max = _mean_max(e1_vals)
    e2_mean, e2_max = _mean_max(e2_vals)
    e3_mean, e3_max = _mean_max(e3_vals)

    denom = pitch_px if np.isfinite(pitch_px) and pitch_px > 1e-6 else float("nan")

    def norm(v: float) -> float:
        if not np.isfinite(v) or not np.isfinite(denom):
            return float("nan")
        return float(v / denom)

    return {
        "per_template": per_template,
        "E1": {
            "mean_px": e1_mean,
            "max_px": e1_max,
            "mean_norm_pitch": norm(e1_mean),
            "max_norm_pitch": norm(e1_max),
            "values_px": e1_vals,
        },
        "E2": {
            "mean_px": e2_mean,
            "max_px": e2_max,
            "mean_norm_pitch": norm(e2_mean),
            "max_norm_pitch": norm(e2_max),
            "values_px": e2_vals,
        },
        "E3": {
            "mean_px": e3_mean,
            "max_px": e3_max,
            "mean_norm_pitch": norm(e3_mean),
            "max_norm_pitch": norm(e3_max),
            "values_px": e3_vals,
        },
    }


def _draw_cross(img: np.ndarray, x: int, y: int, color: Tuple[int, int, int], size: int = 5, th: int = 2) -> None:
    cv2.line(img, (x - size, y), (x + size, y), color, th, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, th, cv2.LINE_AA)


def _draw_threeway_overlay(
    *,
    raw_image: np.ndarray,
    det_points: List[Dict[str, Any]],
    template_ids: List[str],
    fit_points_raw: List[Optional[Tuple[float, float]]],
    slice_points_raw: List[Optional[Tuple[float, float]]],
    blank_idx: Optional[int],
    pitch_px: float,
    matched_count: int,
    pure_filled_ratio: float,
    transform_type: str,
    blank_id: Optional[int],
    blank_confidence: float,
    e1_mean: float,
    e1_max: float,
    e2_mean: float,
    e2_max: float,
    e3_mean: float,
    e3_max: float,
    output_path: Path,
) -> None:
    canvas = raw_image.copy()

    # P_det: blue
    for d in det_points:
        x = _safe_float(d.get("x"))
        y = _safe_float(d.get("y"))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(canvas, (int(round(x)), int(round(y))), 3, (255, 120, 30), -1, cv2.LINE_AA)

    # P_fit: yellow, P_slice_backproj: green
    for i in range(len(template_ids)):
        tid = str(template_ids[i])
        fp = fit_points_raw[i] if i < len(fit_points_raw) else None
        sp = slice_points_raw[i] if i < len(slice_points_raw) else None

        if fp is not None:
            fx, fy = int(round(fp[0])), int(round(fp[1]))
            _draw_cross(canvas, fx, fy, (0, 220, 255), size=6, th=2)
            cv2.putText(canvas, tid, (fx + 5, fy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        if sp is not None:
            sx, sy = int(round(sp[0])), int(round(sp[1]))
            cv2.circle(canvas, (sx, sy), 5, (70, 255, 70), 1, cv2.LINE_AA)
            cv2.putText(canvas, tid, (sx + 5, sy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 255, 70), 1, cv2.LINE_AA)
        if fp is not None and sp is not None:
            cv2.line(
                canvas,
                (int(round(fp[0])), int(round(fp[1]))),
                (int(round(sp[0])), int(round(sp[1]))),
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

    mark_idx = blank_id if blank_id is not None else blank_idx
    if mark_idx is not None and 0 <= int(mark_idx) < len(fit_points_raw):
        pt = fit_points_raw[int(mark_idx)]
        if pt is not None:
            bx, by = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(canvas, (bx, by), 16, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "BLANK", (bx - 26, by - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)

    lines = [
        f"pitch_px={pitch_px:.3f}" if np.isfinite(pitch_px) else "pitch_px=nan",
        f"E1 mean/max={e1_mean:.3f}/{e1_max:.3f}" if np.isfinite(e1_mean) else "E1 mean/max=nan/nan",
        f"E2 mean/max={e2_mean:.3f}/{e2_max:.3f}" if np.isfinite(e2_mean) else "E2 mean/max=nan/nan",
        f"E3 mean/max={e3_mean:.3f}/{e3_max:.3f}" if np.isfinite(e3_mean) else "E3 mean/max=nan/nan",
        f"matched_count={matched_count} pure_filled_ratio={pure_filled_ratio:.3f}" if np.isfinite(pure_filled_ratio) else f"matched_count={matched_count} pure_filled_ratio=nan",
        f"transform={transform_type} blank_id={blank_id} blank_conf={blank_confidence:.3f}" if np.isfinite(blank_confidence) else f"transform={transform_type} blank_id={blank_id} blank_conf=nan",
        "blue=P_det yellow=P_fit green=P_slice_backproj",
    ]

    x0, y0 = 10, 24
    max_w = 0
    for ln in lines:
        (w, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        max_w = max(max_w, w)
    cv2.rectangle(canvas, (x0 - 7, y0 - 20), (x0 + max_w + 10, y0 + 24 * len(lines) + 4), (0, 0, 0), -1)
    cv2.rectangle(canvas, (x0 - 7, y0 - 20), (x0 + max_w + 10, y0 + 24 * len(lines) + 4), (70, 70, 70), 1)

    for i, ln in enumerate(lines):
        cv2.putText(canvas, ln, (x0, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _build_fallback_canvas(
    det_points: List[Dict[str, Any]],
    fit_points_raw: List[Optional[Tuple[float, float]]],
    slice_points_raw: List[Optional[Tuple[float, float]]],
) -> np.ndarray:
    xs: List[float] = []
    ys: List[float] = []
    for d in det_points:
        x = _safe_float(d.get("x"))
        y = _safe_float(d.get("y"))
        if np.isfinite(x) and np.isfinite(y):
            xs.append(float(x))
            ys.append(float(y))
    for p in fit_points_raw + slice_points_raw:
        if p is not None:
            xs.append(float(p[0]))
            ys.append(float(p[1]))

    if not xs or not ys:
        return np.zeros((1024, 1024, 3), dtype=np.uint8)

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    w = int(max(512, min(4096, (max_x - min_x) + 200)))
    h = int(max(512, min(4096, (max_y - min_y) + 200)))
    return np.zeros((h, w, 3), dtype=np.uint8)


def _normalize_failure_reasons(
    run_exception: str,
    topo: Dict[str, Any],
    geom: Dict[str, Any],
) -> str:
    reasons: List[str] = []
    if run_exception:
        reasons.append(run_exception)

    final_reason = topo.get("final_reason") or topo.get("failure_reason")
    if isinstance(final_reason, str) and final_reason:
        reasons.append(final_reason)

    qc = topo.get("qc", {}) if isinstance(topo, dict) else {}
    qc_reasons = qc.get("qc_fail_reasons")
    if isinstance(qc_reasons, list):
        reasons.extend(str(x) for x in qc_reasons if str(x))

    geo_reasons = geom.get("failure_reasons")
    if isinstance(geo_reasons, list):
        reasons.extend(str(x) for x in geo_reasons if str(x))

    # keep order, deduplicate
    seen = set()
    uniq: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return "|".join(uniq)


def _classify_responsibility(row: Dict[str, Any]) -> str:
    e1n = _safe_float(row.get("E1_mean_norm"))
    e2n = _safe_float(row.get("E2_mean_norm"))
    e2maxn = _safe_float(row.get("E2_max_norm"))
    e3n = _safe_float(row.get("E3_mean_norm"))

    if np.isfinite(e2n) and (e2n > 0.10 or (np.isfinite(e2maxn) and e2maxn > 0.20)):
        return "geometry_warp_slicing"
    if np.isfinite(e2n) and e2n <= 0.10 and np.isfinite(e1n) and e1n > 0.10:
        return "topology_fit"
    if (
        np.isfinite(e1n)
        and np.isfinite(e2n)
        and np.isfinite(e3n)
        and e1n <= 0.10
        and e2n <= 0.10
        and e3n > 0.10
    ):
        return "yolo_bbox_center"
    if str(row.get("final_status", "")).lower() != "success":
        return "failed_before_comparable"
    return "mixed_or_good"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage1 canonical alignment and dump three-way diagnostics")
    parser.add_argument("--input-dir", type=Path, required=True, help="Root dir containing detection jsons")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for stage1-post runs")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--template", type=Path, default=Path("configs/templates/pinwheel_v3_centered.json"))
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--min-topology-detections", type=int, default=8)
    parser.add_argument("--enable-fallback-detection", action="store_true", help="Enable fallback detection retry (default: off)")
    parser.add_argument("--dump-threeway-debug", action="store_true", help="Dump per-sample debug_threeway_overlay.png and debug_threeway_metrics.json")
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    stage1_cfg = cfg.stage1.model_copy(deep=True)
    if stage1_cfg.topology is None:
        stage1_cfg.topology = TopologyConfig()
    stage1_cfg.topology.template_path = str(args.template)

    det_jsons = _discover_jsons(args.input_dir)
    if len(det_jsons) == 0:
        raise RuntimeError(f"No detection json found under: {args.input_dir}")

    det_jsons = det_jsons[: max(1, args.max_samples)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    all_e1: List[float] = []
    all_e2: List[float] = []
    all_e3: List[float] = []
    key_cases_root = args.output_dir / "key_cases"
    key_case_count = 0

    for i, det_json in enumerate(det_jsons, start=1):
        chip_id = det_json.parent.name
        print(f"[{i}/{len(det_jsons)}] {chip_id}")

        run_exception = ""
        try:
            run_stage1_postprocess_from_json(
                detections_json_path=det_json,
                output_dir=args.output_dir,
                config=stage1_cfg,
                min_topology_detections=args.min_topology_detections,
                enable_fallback_detection=bool(args.enable_fallback_detection),
                save_individual_slices=False,
                save_debug=True,
            )
        except Exception as e:
            run_exception = str(e)

        run_dir = args.output_dir / chip_id
        det_payload = _safe_load_json(det_json)
        topo = _safe_load_json(run_dir / "debug_stage1_topology.json")
        geom = _safe_load_json(run_dir / "debug_geometry_alignment.json")
        stage_meta = _safe_load_json(run_dir / "stage1_metadata.json")

        # P_det
        det_points = _extract_det_points(topo.get("detections", []))
        if not det_points:
            det_points = _extract_det_points(det_payload.get("detections", []))

        # P_fit + mapping semantics
        template_ids, fit_points_raw, matched_det_idx, pure_filled, detected_by_model = _extract_topology_points(topo, geom)

        # P_slice (backproj to raw)
        slice_points_raw = _extract_slice_backproj_points(geom, len(template_ids))

        qc = topo.get("qc", {}) if isinstance(topo, dict) else {}
        fit_info = topo.get("fit", {}) if isinstance(topo, dict) else {}
        blank = topo.get("blank", {}) if isinstance(topo, dict) else {}

        pitch_px = _safe_float(
            topo.get("pitch_px", qc.get("pitch_px", fit_info.get("pitch_px", float("nan"))))
        )

        matched_count = _safe_int(topo.get("matched_count", qc.get("matched_count", 0)), default=0)
        det_used_count = _safe_int(topo.get("det_used_count", qc.get("det_used_count", 0)), default=0)
        fill_count = _safe_int(topo.get("fill_count", qc.get("fill_count", 0)), default=0)
        reject_det_count = _safe_int(topo.get("reject_det_count", qc.get("reject_det_count", 0)), default=0)
        pure_filled_ratio = _safe_float(topo.get("pure_filled_ratio", qc.get("pure_filled_ratio", float("nan"))))
        det_fit_dist_mean_px = _safe_float(
            topo.get("det_priority", {}).get("det_fit_dist_mean_px", qc.get("det_fit_dist_mean_px", float("nan")))
            if isinstance(topo.get("det_priority"), dict)
            else qc.get("det_fit_dist_mean_px", float("nan"))
        )
        det_fit_dist_max_px = _safe_float(
            topo.get("det_priority", {}).get("det_fit_dist_max_px", qc.get("det_fit_dist_max_px", float("nan")))
            if isinstance(topo.get("det_priority"), dict)
            else qc.get("det_fit_dist_max_px", float("nan"))
        )

        threeway = _build_threeway_metrics(
            template_ids=template_ids,
            det_points=det_points,
            fit_points_raw=fit_points_raw,
            slice_points_raw=slice_points_raw,
            matched_det_idx=matched_det_idx,
            pure_filled=pure_filled,
            detected_by_model=detected_by_model,
            pitch_px=pitch_px,
        )

        e1 = threeway["E1"]
        e2 = threeway["E2"]
        e3 = threeway["E3"]

        all_e1.extend(e1.get("values_px", []))
        all_e2.extend(e2.get("values_px", []))
        all_e3.extend(e3.get("values_px", []))

        transform_type = str(geom.get("transform_type", fit_info.get("transform_type", "none")))
        blank_id = blank.get("blank_id", None)
        blank_id_old = blank.get("blank_id_old", None)
        blank_id_color = blank.get("blank_id_color", None)
        blank_status_color = blank.get("blank_status_color", None)
        blank_mode = blank.get("blank_mode", None)
        reference_arm = blank.get("reference_arm", None)
        blank_conf = _safe_float(blank.get("blank_confidence", qc.get("blank_confidence", float("nan"))))
        blank_scores = blank.get("blank_scores", {}) if isinstance(blank, dict) else {}
        blank_vals = [float(v) for v in blank_scores.values()] if isinstance(blank_scores, dict) else []
        blank_score_spread = float(max(blank_vals) - min(blank_vals)) if len(blank_vals) >= 2 else float("nan")
        blank_is_detected = int(bool(qc.get("blank_is_detected", False)))

        final_status = str(topo.get("final_status", topo.get("status", "failed")))
        failure_reason = _normalize_failure_reasons(run_exception, topo, geom)

        strict_12_slices = int(_read_slice_count(run_dir) == 12)

        threeway_json_path = run_dir / "debug_threeway_metrics.json"
        overlay_path = run_dir / "debug_threeway_overlay.png"

        threeway_payload: Dict[str, Any] = {
            "sample_id": chip_id,
            "raw_image_path": str(_resolve_raw_image_path(run_dir, det_json, det_payload, topo) or ""),
            "detections_json": str(det_json),
            "final_status": final_status,
            "failure_reason": failure_reason,
            "transform_type": transform_type,
            "pitch_px": pitch_px if np.isfinite(pitch_px) else None,
            "template_ids": [str(t) for t in template_ids],
            "P_det": [
                {
                    "det_idx": int(d["det_idx"]),
                    "x": float(d["x"]),
                    "y": float(d["y"]),
                    "conf": None if not np.isfinite(_safe_float(d.get("conf"))) else float(_safe_float(d.get("conf"))),
                    "class_id": int(d.get("class_id", -1)),
                }
                for d in det_points
            ],
            "P_fit_raw": [
                {
                    "template_index": int(i),
                    "template_id": str(template_ids[i]),
                    "x": None if fit_points_raw[i] is None else float(fit_points_raw[i][0]),
                    "y": None if fit_points_raw[i] is None else float(fit_points_raw[i][1]),
                    "matched_det_idx": int(matched_det_idx[i]),
                    "is_pure_filled": bool(pure_filled[i]),
                    "detected_by_model": bool(detected_by_model[i]),
                }
                for i in range(len(template_ids))
            ],
            "P_slice_raw_backproj": [
                {
                    "template_index": int(i),
                    "template_id": str(template_ids[i]),
                    "x": None if slice_points_raw[i] is None else float(slice_points_raw[i][0]),
                    "y": None if slice_points_raw[i] is None else float(slice_points_raw[i][1]),
                }
                for i in range(len(template_ids))
            ],
            "per_template_metrics": threeway["per_template"],
            "summary_metrics": {
                "matched_count": matched_count,
                "det_used_count": det_used_count,
                "fill_count": fill_count,
                "reject_det_count": reject_det_count,
                "pure_filled_ratio": pure_filled_ratio if np.isfinite(pure_filled_ratio) else None,
                "det_fit_dist_mean_px": det_fit_dist_mean_px if np.isfinite(det_fit_dist_mean_px) else None,
                "det_fit_dist_max_px": det_fit_dist_max_px if np.isfinite(det_fit_dist_max_px) else None,
                "E1_mean_px": e1["mean_px"] if np.isfinite(e1["mean_px"]) else None,
                "E1_max_px": e1["max_px"] if np.isfinite(e1["max_px"]) else None,
                "E2_mean_px": e2["mean_px"] if np.isfinite(e2["mean_px"]) else None,
                "E2_max_px": e2["max_px"] if np.isfinite(e2["max_px"]) else None,
                "E3_mean_px": e3["mean_px"] if np.isfinite(e3["mean_px"]) else None,
                "E3_max_px": e3["max_px"] if np.isfinite(e3["max_px"]) else None,
                "E1_mean_norm": e1["mean_norm_pitch"] if np.isfinite(e1["mean_norm_pitch"]) else None,
                "E2_mean_norm": e2["mean_norm_pitch"] if np.isfinite(e2["mean_norm_pitch"]) else None,
                "E3_mean_norm": e3["mean_norm_pitch"] if np.isfinite(e3["mean_norm_pitch"]) else None,
                "E2_max_norm": e2["max_norm_pitch"] if np.isfinite(e2["max_norm_pitch"]) else None,
            },
            "blank": {
                "blank_id": None if blank_id is None else int(blank_id),
                "blank_id_old": None if blank_id_old is None else int(blank_id_old),
                "blank_id_color": None if blank_id_color is None else int(blank_id_color),
                "blank_status_color": None if blank_status_color is None else str(blank_status_color),
                "blank_mode": None if blank_mode is None else str(blank_mode),
                "reference_arm": None if reference_arm is None else str(reference_arm),
                "blank_confidence": blank_conf if np.isfinite(blank_conf) else None,
                "blank_is_detected": bool(blank_is_detected),
                "blank_score_spread": blank_score_spread if np.isfinite(blank_score_spread) else None,
            },
            "quality_gate_passed": stage_meta.get("quality_gate_passed"),
            "gate_failure_reasons": failure_reason.split("|") if failure_reason else [],
            "strict_12_slices": bool(strict_12_slices),
        }

        raw_path = _resolve_raw_image_path(run_dir, det_json, det_payload, topo)
        raw_image = cv2.imread(str(raw_path)) if raw_path is not None else None
        if raw_image is None and args.dump_threeway_debug:
            raw_image = _build_fallback_canvas(det_points, fit_points_raw, slice_points_raw)

        if args.dump_threeway_debug:
            with open(threeway_json_path, "w", encoding="utf-8") as f:
                json.dump(threeway_payload, f, ensure_ascii=False, indent=2)

            if raw_image is not None:
                _draw_threeway_overlay(
                    raw_image=raw_image,
                    det_points=det_points,
                    template_ids=template_ids,
                    fit_points_raw=fit_points_raw,
                    slice_points_raw=slice_points_raw,
                    blank_idx=_safe_int(geom.get("blank_idx"), -1),
                    pitch_px=pitch_px,
                    matched_count=matched_count,
                    pure_filled_ratio=pure_filled_ratio,
                    transform_type=transform_type,
                    blank_id=None if blank_id is None else int(blank_id),
                    blank_confidence=blank_conf,
                    e1_mean=e1["mean_px"],
                    e1_max=e1["max_px"],
                    e2_mean=e2["mean_px"],
                    e2_max=e2["max_px"],
                    e3_mean=e3["mean_px"],
                    e3_max=e3["max_px"],
                    output_path=overlay_path,
                )

        row: Dict[str, Any] = {
            "sample_id": chip_id,
            "detections_json": str(det_json),
            "run_dir": str(run_dir),
            "final_status": final_status,
            "failure_reason": failure_reason,
            "gate_failure_reasons": failure_reason,
            "transform_type": transform_type,
            "pitch_px": pitch_px,
            "matched_count": matched_count,
            "det_used_count": det_used_count,
            "fill_count": fill_count,
            "reject_det_count": reject_det_count,
            "pure_filled_ratio": pure_filled_ratio,
            "det_fit_dist_mean_px": det_fit_dist_mean_px,
            "det_fit_dist_max_px": det_fit_dist_max_px,
            "E1_mean": e1["mean_px"],
            "E1_max": e1["max_px"],
            "E2_mean": e2["mean_px"],
            "E2_max": e2["max_px"],
            "E3_mean": e3["mean_px"],
            "E3_max": e3["max_px"],
            "E1_mean_norm": e1["mean_norm_pitch"],
            "E2_mean_norm": e2["mean_norm_pitch"],
            "E3_mean_norm": e3["mean_norm_pitch"],
            "E2_max_norm": e2["max_norm_pitch"],
            "blank_id": blank_id,
            "blank_id_old": blank_id_old,
            "blank_id_color": blank_id_color,
            "blank_status_color": blank_status_color,
            "blank_mode": blank_mode,
            "reference_arm": reference_arm,
            "blank_confidence": blank_conf,
            "blank_score_spread": blank_score_spread,
            "blank_is_detected": blank_is_detected,
            "strict_12_slices": strict_12_slices,
            "stage1_quality_gate_passed": stage_meta.get("quality_gate_passed"),
            "threeway_debug_json": str(threeway_json_path) if args.dump_threeway_debug else "",
            "threeway_overlay": str(overlay_path) if args.dump_threeway_debug and overlay_path.exists() else "",
        }
        row["responsibility_segment"] = _classify_responsibility(row)
        rows.append(row)

        changed_blank = (blank_id_old != blank_id_color)
        if changed_blank:
            key_case_dir = key_cases_root / chip_id
            key_case_dir.mkdir(parents=True, exist_ok=True)
            key_case_count += 1
            for src_name in (
                "debug_stage1_topology.png",
                "debug_stage1_topology.json",
                "debug_overlay.png",
                "debug_threeway_overlay.png",
                "debug_threeway_metrics.json",
                "stage1_metadata.json",
            ):
                src = run_dir / src_name
                if src.exists():
                    shutil.copy2(src, key_case_dir / src_name)
            if raw_path is not None and raw_path.exists():
                shutil.copy2(raw_path, key_case_dir / raw_path.name)

    if not rows:
        raise RuntimeError("No rows generated")

    summary_path = args.output_dir / "threeway_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # keep backward-compatible filename
    legacy_path = args.output_dir / "summary.csv"
    with open(legacy_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    e1_p50, e1_p90, e1_p95 = _quantiles(all_e1)
    e2_p50, e2_p90, e2_p95 = _quantiles(all_e2)
    e3_p50, e3_p90, e3_p95 = _quantiles(all_e3)

    success_cnt = sum(1 for r in rows if str(r.get("final_status", "")).lower() == "success")
    strict_cnt = sum(int(r.get("strict_12_slices", 0)) for r in rows)

    seg_counts: Dict[str, int] = {}
    for r in rows:
        k = str(r.get("responsibility_segment", "unknown"))
        seg_counts[k] = seg_counts.get(k, 0) + 1
    major_segment = max(seg_counts.items(), key=lambda kv: kv[1])[0] if seg_counts else "unknown"

    print("")
    print(f"Done. threeway summary: {summary_path}")
    print(f"Success: {success_cnt}/{len(rows)}")
    print(f"Strict 12 slices: {strict_cnt}/{len(rows)}")
    print(
        "E1 quantiles px (p50/p90/p95): "
        f"{e1_p50:.3f}/{e1_p90:.3f}/{e1_p95:.3f}"
    )
    print(
        "E2 quantiles px (p50/p90/p95): "
        f"{e2_p50:.3f}/{e2_p90:.3f}/{e2_p95:.3f}"
    )
    print(
        "E3 quantiles px (p50/p90/p95): "
        f"{e3_p50:.3f}/{e3_p90:.3f}/{e3_p95:.3f}"
    )
    print(f"Responsibility segments: {seg_counts}")
    print(f"Main responsibility segment: {major_segment}")
    print(f"Blank-changed key cases: {key_case_count} (saved to {key_cases_root})")


if __name__ == "__main__":
    main()
