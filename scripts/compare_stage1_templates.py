#!/usr/bin/env python3
"""
Compare Stage1 topology postprocess stability between:
1) Default cross template (template_path = None)
2) User pinwheel template (template_path = custom JSON)

Outputs:
- template_compare_summary.csv
- template_compare_report.md
- per-sample overlays and debug artifacts from stage1-post runs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microfluidics_chip.core.config import (  # noqa: E402
    Stage1Config,
    TopologyConfig,
    load_config_from_yaml,
)
from microfluidics_chip.pipelines.stage1 import run_stage1_postprocess_from_json  # noqa: E402


@dataclass
class SampleRecord:
    sample_id: str
    detections_json: Path
    raw_image: Path
    det_count: int
    scale_proxy_px: float
    brightness_mean: float
    brightness_std: float
    orientation_deg: float


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _resolve_raw_image(detections_json: Path, payload: Dict[str, Any]) -> Optional[Path]:
    raw_rel = payload.get("raw_image_path")
    candidates: List[Path] = []
    if isinstance(raw_rel, str) and raw_rel:
        candidates.append(detections_json.parent / raw_rel)
    candidates.append(detections_json.parent / "raw.png")
    for p in candidates:
        if p.exists():
            return p
    return None


def _estimate_nn_pitch(centers: np.ndarray) -> float:
    if centers.shape[0] < 4:
        return float("nan")
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = np.min(d, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return float("nan")
    return float(np.median(nn))


def _estimate_orientation_deg(centers: np.ndarray) -> float:
    if centers.shape[0] < 2:
        return float("nan")
    c = centers - np.mean(centers, axis=0, keepdims=True)
    cov = np.cov(c.T)
    vals, vecs = np.linalg.eigh(cov)
    main = vecs[:, np.argmax(vals)]
    return float(np.degrees(np.arctan2(main[1], main[0])))


def _discover_jsons(root: Path) -> List[Path]:
    patterns = [
        "adaptive_yolo_raw_detections.json",
        "yolo_raw_detections.json",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    return sorted(set(files))


def collect_samples(input_root: Path) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for p in _discover_jsons(input_root):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        detections = payload.get("detections", [])
        if not isinstance(detections, list) or len(detections) < 2:
            continue
        raw_path = _resolve_raw_image(p, payload)
        if raw_path is None:
            continue
        img = cv2.imread(str(raw_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        centers = []
        for d in detections:
            c = d.get("center")
            if isinstance(c, (list, tuple)) and len(c) == 2:
                centers.append([float(c[0]), float(c[1])])
        if len(centers) < 2:
            continue
        centers_np = np.array(centers, dtype=np.float32)
        nn_pitch = _estimate_nn_pitch(centers_np)
        if not np.isfinite(nn_pitch):
            roi = payload.get("roi_bbox")
            if isinstance(roi, (list, tuple)) and len(roi) == 4:
                nn_pitch = max(8.0, min(float(roi[2] - roi[0]), float(roi[3] - roi[1])) / 6.0)
            else:
                nn_pitch = 50.0
        rec = SampleRecord(
            sample_id=p.parent.name,
            detections_json=p,
            raw_image=raw_path,
            det_count=len(detections),
            scale_proxy_px=float(nn_pitch),
            brightness_mean=float(np.mean(gray)),
            brightness_std=float(np.std(gray)),
            orientation_deg=_estimate_orientation_deg(centers_np),
        )
        records.append(rec)
    return records


def _normalize_feature_column(col: np.ndarray) -> np.ndarray:
    mask = np.isfinite(col)
    if not np.any(mask):
        return np.zeros_like(col)
    fill = float(np.median(col[mask]))
    c = col.copy()
    c[~mask] = fill
    std = float(np.std(c))
    if std < 1e-6:
        return c * 0.0
    return (c - float(np.mean(c))) / std


def select_diverse_samples(records: List[SampleRecord], n: int) -> List[SampleRecord]:
    if len(records) <= n:
        return records

    feats = np.array([
        [r.scale_proxy_px, r.brightness_mean, r.brightness_std, abs(r.orientation_deg), float(r.det_count)]
        for r in records
    ], dtype=np.float32)
    norm = np.column_stack([_normalize_feature_column(feats[:, i]) for i in range(feats.shape[1])])

    selected: List[int] = []
    seed_indices: List[int] = []
    for col_idx in range(norm.shape[1]):
        seed_indices.append(int(np.argmin(norm[:, col_idx])))
        seed_indices.append(int(np.argmax(norm[:, col_idx])))
    for idx in seed_indices:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= n:
            break

    while len(selected) < n:
        best_idx = None
        best_dist = -1.0
        sel = norm[selected]
        for i in range(len(records)):
            if i in selected:
                continue
            d = np.linalg.norm(sel - norm[i], axis=1)
            min_d = float(np.min(d))
            if min_d > best_dist:
                best_dist = min_d
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)

    selected = sorted(selected[:n], key=lambda i: records[i].sample_id)
    return [records[i] for i in selected]


def _load_stage1_config(config_path: Path) -> Stage1Config:
    cfg = load_config_from_yaml(config_path)
    return cfg.stage1.model_copy(deep=True)


def _build_stage1_config(base_stage1: Stage1Config, template_path: Optional[Path]) -> Stage1Config:
    cfg = base_stage1.model_copy(deep=True)
    if cfg.topology is None:
        cfg.topology = TopologyConfig()
    cfg.topology.template_path = str(template_path) if template_path is not None else None
    return cfg


def _read_debug_json(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "debug_stage1_topology.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _draw_cross(img: np.ndarray, x: int, y: int, color: Tuple[int, int, int], size: int = 6, thickness: int = 2) -> None:
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def build_debug_overlay(debug_payload: Dict[str, Any], approx_pitch_mm: float, output_path: Path) -> Optional[Path]:
    raw_path = debug_payload.get("raw_image_path")
    if not raw_path:
        return None
    raw_img = cv2.imread(str(raw_path))
    if raw_img is None:
        return None

    canvas = raw_img.copy()

    # YOLO detections: blue
    for d in debug_payload.get("detections", []):
        x = int(round(_safe_float(d.get("x"), 0.0)))
        y = int(round(_safe_float(d.get("y"), 0.0)))
        cv2.circle(canvas, (x, y), 5, (255, 80, 30), -1, cv2.LINE_AA)

    # Fitted points: green/yellow
    pure_color = (0, 255, 255)
    fit_color = (80, 255, 80)
    for fp in debug_payload.get("fitted_points", []):
        x = int(round(_safe_float(fp.get("x"), 0.0)))
        y = int(round(_safe_float(fp.get("y"), 0.0)))
        idx = fp.get("template_index", -1)
        pure_filled = bool(fp.get("pure_filled", False))
        col = pure_color if pure_filled else fit_color
        _draw_cross(canvas, x, y, col, size=6, thickness=2)
        cv2.putText(
            canvas,
            str(idx),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )

    blank = debug_payload.get("blank", {})
    outer_idxs = blank.get("candidate_indices", [])
    blank_id = blank.get("blank_id")
    point_map = {int(fp.get("template_index", -1)): fp for fp in debug_payload.get("fitted_points", [])}

    # Outermost candidates: magenta rings
    for idx in outer_idxs:
        if int(idx) not in point_map:
            continue
        fp = point_map[int(idx)]
        x = int(round(_safe_float(fp.get("x"), 0.0)))
        y = int(round(_safe_float(fp.get("y"), 0.0)))
        cv2.circle(canvas, (x, y), 12, (255, 0, 255), 2, cv2.LINE_AA)

    # Final blank: red ring + label
    if blank_id is not None and int(blank_id) in point_map:
        fp = point_map[int(blank_id)]
        x = int(round(_safe_float(fp.get("x"), 0.0)))
        y = int(round(_safe_float(fp.get("y"), 0.0)))
        cv2.circle(canvas, (x, y), 18, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, "BLANK", (x - 26, y - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    fit = debug_payload.get("fit", {})
    qc = debug_payload.get("qc", {})
    pitch_px = _safe_float(debug_payload.get("pitch_px", qc.get("pitch_px")))
    px_per_mm = pitch_px / approx_pitch_mm if np.isfinite(pitch_px) and approx_pitch_mm > 0 else float("nan")

    overlay_lines = [
        f"status={debug_payload.get('final_status', debug_payload.get('status'))}",
        f"rmse_px={_safe_float(fit.get('rmse_px', qc.get('rmse_px'))):.3f}",
        f"matched={int(debug_payload.get('matched_count', qc.get('matched_count', 0)))}",
        f"inliers={int(fit.get('n_inliers', qc.get('n_inliers', 0)))}",
        f"pure_filled_ratio={_safe_float(debug_payload.get('pure_filled_ratio', qc.get('pure_filled_ratio'))):.3f}",
        f"blank_conf={_safe_float(blank.get('blank_confidence', qc.get('blank_confidence'))):.3f}",
        f"px_per_mm={px_per_mm:.3f}",
    ]

    y0 = 24
    for line in overlay_lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (10, y0 - th - 6), (16 + tw, y0 + 4), (0, 0, 0), -1)
        cv2.putText(canvas, line, (13, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 255, 230), 2, cv2.LINE_AA)
        y0 += 26

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def _count_assignment_issues(fitted_points: List[Dict[str, Any]]) -> Tuple[int, int]:
    cnt = Counter()
    mismatch = 0
    for fp in fitted_points:
        det_idx = fp.get("matched_det_idx", -1)
        detected = bool(fp.get("detected_by_model", False))
        pure = bool(fp.get("pure_filled", False))
        if isinstance(det_idx, int) and det_idx >= 0:
            cnt[det_idx] += 1
        if pure and det_idx != -1:
            mismatch += 1
        if (not detected) and det_idx != -1:
            mismatch += 1
    dup = sum(v - 1 for v in cnt.values() if v > 1)
    return dup, mismatch


def extract_metrics(
    sample: SampleRecord,
    run_name: str,
    debug_payload: Dict[str, Any],
    debug_json_path: Path,
    overlay_path: Optional[Path],
    approx_pitch_mm: float,
) -> Dict[str, Any]:
    fit = debug_payload.get("fit", {})
    qc = debug_payload.get("qc", {})
    blank = debug_payload.get("blank", {})
    fp = debug_payload.get("fitted_points", [])

    pitch_px = _safe_float(debug_payload.get("pitch_px", qc.get("pitch_px")))
    px_per_mm = pitch_px / approx_pitch_mm if np.isfinite(pitch_px) and approx_pitch_mm > 0 else float("nan")

    blank_scores = blank.get("blank_scores", {})
    score_vals = [float(v) for v in blank_scores.values()] if isinstance(blank_scores, dict) else []
    blank_score_spread = float(max(score_vals) - min(score_vals)) if len(score_vals) >= 2 else float("nan")

    final_status = debug_payload.get("final_status", debug_payload.get("status", "unknown"))
    quality_gate_passed = 1 if str(final_status).lower() == "success" else 0
    fit_success = int(bool(qc.get("fit_success", fit.get("fit_success", False))))

    dup_count, mismatch_count = _count_assignment_issues(fp)

    row: Dict[str, Any] = {
        "sample_id": sample.sample_id,
        "run": run_name,
        "detections_json": str(sample.detections_json),
        "raw_image": str(sample.raw_image),
        "debug_json": str(debug_json_path),
        "debug_overlay": str(overlay_path) if overlay_path else "",
        "status": debug_payload.get("status"),
        "final_status": final_status,
        "quality_gate_passed": quality_gate_passed,
        "fit_success": fit_success,
        "transform_type": fit.get("transform_type"),
        "n_det": int(debug_payload.get("n_det", len(debug_payload.get("detections", [])))),
        "matched_count": int(debug_payload.get("matched_count", qc.get("matched_count", 0))),
        "n_inliers": int(fit.get("n_inliers", qc.get("n_inliers", 0))),
        "rmse_px": _safe_float(fit.get("rmse_px", qc.get("rmse_px"))),
        "coverage_arms": _safe_float(qc.get("coverage_arms", fit.get("coverage_arms"))),
        "matched_coverage_arms": _safe_float(debug_payload.get("matched_coverage_arms", qc.get("matched_coverage_arms"))),
        "pure_filled_ratio": _safe_float(debug_payload.get("pure_filled_ratio", qc.get("pure_filled_ratio"))),
        "pitch_px": pitch_px,
        "match_thresh_px": _safe_float(debug_payload.get("match_thresh_px", qc.get("match_thresh_px"))),
        "inlier_thresh_px": _safe_float(debug_payload.get("inlier_thresh_px", qc.get("inlier_thresh_px"))),
        "assignment_method": debug_payload.get("assignment_method", qc.get("assignment_method")),
        "px_per_mm": px_per_mm,
        "blank_id": blank.get("blank_id"),
        "blank_is_detected": int(bool(qc.get("blank_is_detected", False))),
        "blank_confidence": _safe_float(blank.get("blank_confidence", qc.get("blank_confidence"))),
        "blank_score_spread": blank_score_spread,
        "failure_reason": debug_payload.get("failure_reason", debug_payload.get("final_reason", "")),
        "qc_fail_reasons": "|".join(qc.get("qc_fail_reasons", [])) if isinstance(qc.get("qc_fail_reasons"), list) else "",
        "duplicate_matched_det_idx_count": int(dup_count),
        "purefilled_semantic_mismatch_count": int(mismatch_count),
    }
    return row


def _safe_mean(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _safe_std(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return float("nan")
    return float(statistics.pstdev(vals))


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _create_ab_collage(left_img: Path, right_img: Path, out_img: Path, left_label: str, right_label: str) -> Optional[Path]:
    a = cv2.imread(str(left_img))
    b = cv2.imread(str(right_img))
    if a is None or b is None:
        return None
    if a.shape[0] != b.shape[0]:
        scale = a.shape[0] / max(1, b.shape[0])
        b = cv2.resize(b, (int(round(b.shape[1] * scale)), a.shape[0]))
    banner_h = 48
    banner = np.zeros((banner_h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
    banner[:] = (25, 25, 25)
    cv2.putText(banner, left_label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 150), 2, cv2.LINE_AA)
    cv2.putText(
        banner,
        right_label,
        (a.shape[1] + 16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (150, 220, 255),
        2,
        cv2.LINE_AA,
    )
    combo = np.vstack([banner, np.hstack([a, b])])
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), combo)
    return out_img


def _aggregate_by_run(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["run"]].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for run, rr in grouped.items():
        out[run] = {
            "n_samples": len(rr),
            "gate_pass_rate": _safe_mean([r["quality_gate_passed"] for r in rr]),
            "fit_success_rate": _safe_mean([r["fit_success"] for r in rr]),
            "rmse_mean": _safe_mean([r["rmse_px"] for r in rr]),
            "rmse_std": _safe_std([r["rmse_px"] for r in rr]),
            "pure_filled_ratio_mean": _safe_mean([r["pure_filled_ratio"] for r in rr]),
            "matched_count_mean": _safe_mean([r["matched_count"] for r in rr]),
            "blank_detect_rate": _safe_mean([r["blank_is_detected"] for r in rr]),
            "px_per_mm_std": _safe_std([r["px_per_mm"] for r in rr]),
            "dup_match_total": int(sum(int(r["duplicate_matched_det_idx_count"]) for r in rr)),
            "semantic_mismatch_total": int(sum(int(r["purefilled_semantic_mismatch_count"]) for r in rr)),
        }
    return out


def _recommendation(cross: Dict[str, Any], pin: Dict[str, Any]) -> Tuple[str, str]:
    pass_gain = _safe_float(pin.get("gate_pass_rate")) - _safe_float(cross.get("gate_pass_rate"))
    rmse_gain = _safe_float(cross.get("rmse_mean")) - _safe_float(pin.get("rmse_mean"))
    pure_gain = _safe_float(cross.get("pure_filled_ratio_mean")) - _safe_float(pin.get("pure_filled_ratio_mean"))

    if pass_gain >= 0.15 and (rmse_gain >= 0.5 or pure_gain >= 0.05):
        return "强替换", "pinwheel 在成功率与几何质量上均有显著改善，建议替换默认模板。"
    if pass_gain > 0.0 or rmse_gain > 0.2 or pure_gain > 0.02:
        return "条件替换", "pinwheel 有改善但不稳定，建议在低质量场景/QC触发时启用。"
    return "不替换", "未观察到显著提升，当前瓶颈更可能在检测质量、ROI与阈值。"


def write_report(
    rows: List[Dict[str, Any]],
    report_path: Path,
    approx_pitch_mm: float,
    template_raw: Path,
    template_centered: Path,
    selected_samples: List[SampleRecord],
    key_collages: List[Path],
) -> None:
    agg = _aggregate_by_run(rows)
    cross = agg.get("cross", {})
    pin = agg.get("pinwheel", {})
    verdict, verdict_reason = _recommendation(cross, pin)

    # failure reasons
    fail_counter: Dict[str, Counter] = {"cross": Counter(), "pinwheel": Counter()}
    for r in rows:
        run = r["run"]
        reasons = str(r.get("qc_fail_reasons", ""))
        if reasons:
            for item in reasons.split("|"):
                if item:
                    fail_counter[run][item] += 1

    lines: List[str] = []
    lines.append("# Stage1 Template Comparison Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Sample count: {len(selected_samples)}")
    lines.append(f"- approx_pitch_mm: {approx_pitch_mm}")
    lines.append(f"- Default template: `DEFAULT_CROSS_TEMPLATE`")
    lines.append(f"- Pinwheel template (raw): `{template_raw}`")
    lines.append(f"- Pinwheel template (centered): `{template_centered}`")
    lines.append("")
    lines.append("## Centering Note")
    lines.append("- Raw point centroid offset: mean_x=0.0689 mm, mean_y=-0.00025 mm.")
    lines.append("- `pinwheel_v3_centered.json` subtracts this centroid to enforce template centered at (0,0).")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Run | N | Gate Pass Rate | Fit Success Rate | RMSE Mean(px) | Pure Filled Ratio Mean | Matched Count Mean | Blank Detect Rate | px_per_mm Std | Duplicate Match Total | Semantic Mismatch Total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in ["cross", "pinwheel"]:
        a = agg.get(run, {})
        lines.append(
            f"| {run} | {a.get('n_samples', 0)} | {a.get('gate_pass_rate', float('nan')):.3f} | "
            f"{a.get('fit_success_rate', float('nan')):.3f} | {a.get('rmse_mean', float('nan')):.3f} | "
            f"{a.get('pure_filled_ratio_mean', float('nan')):.3f} | {a.get('matched_count_mean', float('nan')):.3f} | "
            f"{a.get('blank_detect_rate', float('nan')):.3f} | {a.get('px_per_mm_std', float('nan')):.3f} | "
            f"{a.get('dup_match_total', 0)} | {a.get('semantic_mismatch_total', 0)} |"
        )
    lines.append("")
    lines.append("## Failure Reasons (Top)")
    for run in ["cross", "pinwheel"]:
        lines.append(f"- {run}: {dict(fail_counter[run].most_common(8))}")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- Verdict: **{verdict}**")
    lines.append(f"- Reason: {verdict_reason}")
    lines.append("")
    lines.append("## Key A/B Cases")
    if key_collages:
        for p in key_collages:
            lines.append(f"- `{p}`")
    else:
        lines.append("- No collage generated.")
    lines.append("")
    lines.append("## Geometry Engine Secondary Risk (optional check)")
    lines.append("- `geometry_engine` still uses ideal-cross canonical parameters (`ideal_center_gap`/`ideal_chamber_step`).")
    lines.append("- Even with a better topology template, aligned/cropping can still carry cross-assumption bias.")
    lines.append("- Code reference: `src/microfluidics_chip/stage1_detection/geometry_engine.py`.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_single(
    sample: SampleRecord,
    run_name: str,
    stage1_cfg: Stage1Config,
    output_root: Path,
    approx_pitch_mm: float,
) -> Dict[str, Any]:
    run_output = output_root / "runs" / run_name
    run_output.mkdir(parents=True, exist_ok=True)

    try:
        run_stage1_postprocess_from_json(
            detections_json_path=sample.detections_json,
            output_dir=run_output,
            config=stage1_cfg,
            raw_image_path=None,
            chip_id=sample.sample_id,
            min_topology_detections=None,
            enable_fallback_detection=False,
            save_individual_slices=False,
            save_debug=True,
        )
    except Exception:
        # Failure is expected for some hard samples; debug json is still useful.
        pass

    run_dir = run_output / sample.sample_id
    debug_json_path = run_dir / "debug_stage1_topology.json"
    payload = _read_debug_json(run_dir)
    if payload is None:
        payload = {
            "status": "failed",
            "final_status": "failed",
            "failure_reason": "missing_debug_json",
            "n_det": sample.det_count,
            "fit": {},
            "qc": {},
            "blank": {},
            "fitted_points": [],
            "detections": [],
            "raw_image_path": str(sample.raw_image),
        }
    overlay_path = build_debug_overlay(payload, approx_pitch_mm, run_dir / "debug_overlay.png")
    return extract_metrics(
        sample=sample,
        run_name=run_name,
        debug_payload=payload,
        debug_json_path=debug_json_path,
        overlay_path=overlay_path,
        approx_pitch_mm=approx_pitch_mm,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Stage1 default cross vs pinwheel template")
    parser.add_argument("--input-root", type=Path, default=Path("data/experiments/stage1_yolo_adaptive"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--pinwheel-template", type=Path, default=Path("configs/templates/pinwheel_v3_centered.json"))
    parser.add_argument("--pinwheel-template-raw", type=Path, default=Path("configs/templates/pinwheel_v3_raw.json"))
    parser.add_argument("--out-root", type=Path, default=Path("data/experiments/template_compare"))
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    base_stage1 = _load_stage1_config(args.config)

    # read approx_pitch_mm from template metadata
    tpl_meta = json.loads(args.pinwheel_template.read_text(encoding="utf-8"))
    approx_pitch_mm = _safe_float(
        tpl_meta.get("geometry", {}).get("approx_pitch_mm", 3.0),
        default=3.0,
    )

    records = collect_samples(args.input_root)
    if len(records) == 0:
        raise RuntimeError(f"No valid detection jsons found under: {args.input_root}")
    selected = select_diverse_samples(records, args.num_samples)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.out_root / f"compare_{ts}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Save selected samples metadata
    selected_rows = []
    for s in selected:
        selected_rows.append({
            "sample_id": s.sample_id,
            "detections_json": str(s.detections_json),
            "raw_image": str(s.raw_image),
            "det_count": s.det_count,
            "scale_proxy_px": s.scale_proxy_px,
            "brightness_mean": s.brightness_mean,
            "brightness_std": s.brightness_std,
            "orientation_deg": s.orientation_deg,
        })
    write_csv(selected_rows, output_root / "selected_samples.csv")

    cfg_cross = _build_stage1_config(base_stage1, template_path=None)
    cfg_pin = _build_stage1_config(base_stage1, template_path=args.pinwheel_template)

    all_rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] sample={sample.sample_id}")
        row_a = run_single(sample, "cross", cfg_cross, output_root, approx_pitch_mm)
        row_b = run_single(sample, "pinwheel", cfg_pin, output_root, approx_pitch_mm)
        all_rows.extend([row_a, row_b])

    summary_csv = output_root / "template_compare_summary.csv"
    write_csv(all_rows, summary_csv)

    # Build key-case collages
    by_sample: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in all_rows:
        by_sample[r["sample_id"]][r["run"]] = r

    scored_cases: List[Tuple[float, str]] = []
    for sid, rr in by_sample.items():
        if "cross" not in rr or "pinwheel" not in rr:
            continue
        c = rr["cross"]
        p = rr["pinwheel"]
        pass_gain = float(p["quality_gate_passed"]) - float(c["quality_gate_passed"])
        rmse_gain = _safe_float(c["rmse_px"]) - _safe_float(p["rmse_px"])
        pure_gain = _safe_float(c["pure_filled_ratio"]) - _safe_float(p["pure_filled_ratio"])
        score = pass_gain * 10.0 + rmse_gain + pure_gain * 2.0
        scored_cases.append((score, sid))
    scored_cases.sort(reverse=True)
    key_case_ids = [sid for _, sid in scored_cases[:5]]

    key_collages: List[Path] = []
    for sid in key_case_ids:
        c = by_sample[sid]["cross"]
        p = by_sample[sid]["pinwheel"]
        c_img = Path(c["debug_overlay"])
        p_img = Path(p["debug_overlay"])
        out_img = output_root / "key_cases" / f"{sid}_ab.png"
        built = _create_ab_collage(c_img, p_img, out_img, "A: default-cross", "B: pinwheel")
        if built is not None:
            key_collages.append(built)

    write_report(
        rows=all_rows,
        report_path=output_root / "template_compare_report.md",
        approx_pitch_mm=approx_pitch_mm,
        template_raw=args.pinwheel_template_raw,
        template_centered=args.pinwheel_template,
        selected_samples=selected,
        key_collages=key_collages,
    )

    print("")
    print("Done.")
    print(f"- Summary CSV: {summary_csv}")
    print(f"- Report MD : {output_root / 'template_compare_report.md'}")
    print(f"- Output dir: {output_root}")


if __name__ == "__main__":
    main()

