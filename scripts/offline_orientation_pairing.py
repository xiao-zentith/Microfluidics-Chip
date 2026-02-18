#!/usr/bin/env python3
"""
Offline Stage1.5 orientation pairing.

Given Stage1 canonical slices and ideal GT canonical slices, run a 4-way
orientation race (0/90/180/270) and export best mapping with margin score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ARM_GROUPS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),   # Right arm in pinwheel_v3_centered template order
    (3, 4, 5),   # Down
    (6, 7, 8),   # Left
    (9, 10, 11),  # Up
)
ORIENTATIONS_DEG: Tuple[int, ...] = (0, 90, 180, 270)
REF_ARM_TO_GROUP = {"Right": 0, "Down": 1, "Left": 2, "Up": 3}


@dataclass
class PairResult:
    sample_id: str
    gt_id: str
    chosen_orientation: int
    mapping_pred_to_gt: List[int]
    score_best: float
    score_second: float
    margin: float
    paired_success: bool


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_slices(npz_path: Path, key: str = "slices") -> Optional[np.ndarray]:
    if not npz_path.exists():
        return None
    try:
        payload = np.load(npz_path)
        if key in payload:
            arr = payload[key]
        elif len(payload.files) > 0:
            arr = payload[payload.files[0]]
        else:
            return None
        arr = np.asarray(arr)
        if arr.ndim != 4 or arr.shape[0] < 12:
            return None
        return arr[:12]
    except Exception:
        return None


def _annulus_mask(h: int, w: int, inner_ratio: float = 0.22, outer_ratio: float = 0.47) -> np.ndarray:
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    yy, xx = np.ogrid[:h, :w]
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    base = float(min(h, w))
    r_in = max(2.0, inner_ratio * base)
    r_out = max(r_in + 2.0, outer_ratio * base)
    return (rr2 >= (r_in ** 2)) & (rr2 <= (r_out ** 2))


def _slice_feature(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        return np.zeros((6,), dtype=np.float32)
    h, w = img_bgr.shape[:2]
    mask = _annulus_mask(h, w)
    if int(np.sum(mask)) < 8:
        mask = np.ones((h, w), dtype=bool)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    r = float(np.mean(rgb[..., 0][mask]))
    g = float(np.mean(rgb[..., 1][mask]))
    b = float(np.mean(rgb[..., 2][mask]))
    h_mean = float(np.mean(hsv[..., 0][mask]))
    s_mean = float(np.mean(hsv[..., 1][mask]))
    rg = float(math.log((r + 1e-6) / (g + 1e-6)))
    feat = np.asarray([r / 255.0, g / 255.0, b / 255.0, rg, h_mean / 180.0, s_mean / 255.0], dtype=np.float32)
    return feat


def _extract_features(slices: np.ndarray) -> np.ndarray:
    return np.stack([_slice_feature(slices[i]) for i in range(12)], axis=0).astype(np.float32)


def _orientation_gt_to_pred(shift: int) -> List[int]:
    out: List[int] = []
    for g in range(4):
        src_g = (g + shift) % 4
        for p in range(3):
            out.append(int(src_g * 3 + p))
    return out


def _invert_mapping_gt_to_pred(gt_to_pred: Sequence[int]) -> List[int]:
    pred_to_gt = [-1] * 12
    for gt_idx, pred_idx in enumerate(gt_to_pred):
        if 0 <= int(pred_idx) < 12:
            pred_to_gt[int(pred_idx)] = int(gt_idx)
    return pred_to_gt


def _reference_arm_indices(meta: Dict[str, Any]) -> Optional[List[int]]:
    qm = meta.get("quality_metrics", {}) if isinstance(meta.get("quality_metrics"), dict) else {}
    arm = qm.get("reference_arm_pred", qm.get("reference_arm"))
    if arm is None:
        return None
    arm_key = str(arm).strip()
    group_idx = REF_ARM_TO_GROUP.get(arm_key)
    if group_idx is None:
        return None
    return list(ARM_GROUPS[group_idx])


def _score_orientation(
    pred_feat: np.ndarray,
    gt_feat: np.ndarray,
    gt_to_pred: Sequence[int],
    ref_arm_indices: Optional[Sequence[int]],
    ref_arm_weight: float,
) -> float:
    oriented_pred = pred_feat[np.asarray(gt_to_pred, dtype=np.int32)]
    per_slice = np.linalg.norm(oriented_pred - gt_feat, axis=1)
    weights = np.ones((12,), dtype=np.float32)
    if ref_arm_indices is not None:
        for idx in ref_arm_indices:
            if 0 <= int(idx) < 12:
                weights[int(idx)] = max(1.0, float(ref_arm_weight))
    return float(np.average(per_slice, weights=weights))


def _gather_stage1_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "chamber_slices.npz").exists()]
    return out


def _build_match_index(gt_dirs: Sequence[Path], mode: str, delim: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in gt_dirs:
        name = p.name
        key = name if mode == "exact" else name.split(delim)[0]
        if key not in out:
            out[key] = p
    return out


def _make_preview(
    pred_slices: np.ndarray,
    gt_slices: np.ndarray,
    gt_to_pred: Sequence[int],
    score_lines: Sequence[str],
    out_path: Path,
) -> None:
    tile = 64
    pred_ordered = pred_slices[np.asarray(gt_to_pred, dtype=np.int32)]
    pred_row = np.hstack([cv2.resize(pred_ordered[i], (tile, tile), interpolation=cv2.INTER_AREA) for i in range(12)])
    gt_row = np.hstack([cv2.resize(gt_slices[i], (tile, tile), interpolation=cv2.INTER_AREA) for i in range(12)])
    canvas = np.zeros((tile * 2 + 90, tile * 12, 3), dtype=np.uint8)
    canvas[0:tile, :, :] = pred_row
    canvas[tile:tile * 2, :, :] = gt_row
    cv2.putText(canvas, "Pred(oriented)", (8, tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (8, tile * 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    y0 = tile * 2 + 20
    for i, ln in enumerate(score_lines):
        cv2.putText(canvas, ln, (8, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 240, 200), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Stage1.5 orientation pairing (4-way race).")
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-mode", choices=["exact", "prefix"], default="exact")
    parser.add_argument("--id-delim", type=str, default="__")
    parser.add_argument("--margin-thr", type=float, default=0.02)
    parser.add_argument("--reference-arm-weight", type=float, default=1.5)
    parser.add_argument(
        "--single-gt",
        action="store_true",
        help="Use the only GT sample under gt-root for all pred samples (chip-internal pairing).",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--key-cases-limit", type=int, default=30)
    parser.add_argument("--slice-key", type=str, default="slices")
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    pred_dirs = _gather_stage1_dirs(args.pred_root.resolve())
    gt_dirs = _gather_stage1_dirs(args.gt_root.resolve())
    gt_index = _build_match_index(gt_dirs, mode=args.match_mode, delim=args.id_delim)
    single_gt_dir: Optional[Path] = None
    if bool(args.single_gt):
        if len(gt_dirs) != 1:
            raise RuntimeError(
                f"--single-gt requires exactly one GT run dir under {args.gt_root}, got {len(gt_dirs)}"
            )
        single_gt_dir = gt_dirs[0]

    if args.max_samples and args.max_samples > 0:
        pred_dirs = pred_dirs[: int(args.max_samples)]

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    failure_bucket_counts: Dict[str, int] = {
        "gt_not_found": 0,
        "npz_invalid_or_missing": 0,
        "low_margin": 0,
    }
    preview_rows: List[Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, List[int], List[str]]] = []

    for pred_dir in pred_dirs:
        sample_id = pred_dir.name
        key = sample_id if args.match_mode == "exact" else sample_id.split(args.id_delim)[0]
        gt_dir = single_gt_dir if single_gt_dir is not None else gt_index.get(key)
        if gt_dir is None:
            failures.append({"sample_id": sample_id, "reason": f"gt_not_found_for_key={key}", "fail_bucket": "gt_not_found"})
            failure_bucket_counts["gt_not_found"] += 1
            continue

        pred_slices = _load_slices(pred_dir / "chamber_slices.npz", key=args.slice_key)
        gt_slices = _load_slices(gt_dir / "chamber_slices.npz", key=args.slice_key)
        if pred_slices is None or gt_slices is None:
            failures.append({"sample_id": sample_id, "reason": "npz_invalid_or_missing", "fail_bucket": "npz_invalid_or_missing"})
            failure_bucket_counts["npz_invalid_or_missing"] += 1
            continue

        pred_feat = _extract_features(pred_slices)
        gt_feat = _extract_features(gt_slices)
        gt_meta = _load_json(gt_dir / "stage1_metadata.json")
        ref_arm_indices = _reference_arm_indices(gt_meta)

        scored: List[Tuple[float, int, List[int], List[int]]] = []
        score_text_parts: List[str] = []
        for shift, deg in enumerate(ORIENTATIONS_DEG):
            gt_to_pred = _orientation_gt_to_pred(shift=shift)
            score = _score_orientation(
                pred_feat=pred_feat,
                gt_feat=gt_feat,
                gt_to_pred=gt_to_pred,
                ref_arm_indices=ref_arm_indices,
                ref_arm_weight=float(args.reference_arm_weight),
            )
            pred_to_gt = _invert_mapping_gt_to_pred(gt_to_pred)
            scored.append((float(score), int(deg), gt_to_pred, pred_to_gt))
            score_text_parts.append(f"{deg}deg={score:.5f}")

        scored.sort(key=lambda x: x[0])
        best_score, best_deg, best_gt_to_pred, best_pred_to_gt = scored[0]
        second_score = float(scored[1][0]) if len(scored) > 1 else float("nan")
        margin = float(second_score - best_score) if np.isfinite(second_score) else float("nan")
        paired_success = bool(np.isfinite(margin) and margin >= float(args.margin_thr))
        if not paired_success:
            failure_bucket_counts["low_margin"] += 1

        rec = PairResult(
            sample_id=sample_id,
            gt_id=gt_dir.name,
            chosen_orientation=int(best_deg),
            mapping_pred_to_gt=[int(x) for x in best_pred_to_gt],
            score_best=float(best_score),
            score_second=float(second_score),
            margin=float(margin),
            paired_success=paired_success,
        )
        row = {
            "sample_id": rec.sample_id,
            "gt_id": rec.gt_id,
            "chosen_orientation": rec.chosen_orientation,
            "mapping_pred_to_gt": json.dumps(rec.mapping_pred_to_gt, ensure_ascii=False),
            "score_best": rec.score_best,
            "score_second": rec.score_second,
            "margin": rec.margin,
            "paired_success": int(rec.paired_success),
        }
        rows.append(row)
        preview_lines = [
            f"sample={sample_id} gt={gt_dir.name}",
            f"best={best_deg}deg score={best_score:.5f} second={second_score:.5f} margin={margin:.5f}",
            f"paired_success={int(paired_success)} thr={args.margin_thr:.5f}",
            " | ".join(score_text_parts),
        ]
        preview_rows.append((margin if np.isfinite(margin) else -1e9, row, pred_slices, gt_slices, best_gt_to_pred, preview_lines))

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows_csv(out_dir / "paired_index.csv", rows)
    _write_rows_csv(out_dir / "pairing_failures.csv", failures)

    n_total = len(rows)
    n_success = int(sum(int(r["paired_success"]) for r in rows))
    margins = [float(r["margin"]) for r in rows if np.isfinite(_safe_float(r.get("margin")))]
    summary = {
        "n_total": int(n_total),
        "n_success": int(n_success),
        "paired_success_rate": float(n_success / n_total) if n_total > 0 else float("nan"),
        "margin_mean": float(np.mean(margins)) if margins else float("nan"),
        "margin_median": float(np.median(margins)) if margins else float("nan"),
        "margin_p50": float(np.percentile(margins, 50)) if margins else float("nan"),
        "margin_p90": float(np.percentile(margins, 90)) if margins else float("nan"),
        "margin_p05": float(np.percentile(margins, 5)) if margins else float("nan"),
        "margin_p95": float(np.percentile(margins, 95)) if margins else float("nan"),
        "margin_threshold": float(args.margin_thr),
        "match_mode": str(args.match_mode),
        "id_delim": str(args.id_delim),
        "reference_arm_weight": float(args.reference_arm_weight),
        "single_gt": int(bool(args.single_gt)),
        "fail_bucket_gt_not_found": int(failure_bucket_counts["gt_not_found"]),
        "fail_bucket_npz_invalid_or_missing": int(failure_bucket_counts["npz_invalid_or_missing"]),
        "fail_bucket_low_margin": int(failure_bucket_counts["low_margin"]),
    }
    (out_dir / "pairing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows_csv(out_dir / "pairing_summary.csv", [summary])

    preview_rows.sort(key=lambda x: x[0])
    keep = preview_rows[: max(0, int(args.key_cases_limit))]
    for _, row, pred_slices, gt_slices, gt_to_pred, lines in keep:
        out_img = out_dir / "key_cases" / f"{row['sample_id']}.png"
        _make_preview(pred_slices=pred_slices, gt_slices=gt_slices, gt_to_pred=gt_to_pred, score_lines=lines, out_path=out_img)

    print(f"[INFO] paired samples: {n_total}, success: {n_success}, rate: {summary['paired_success_rate']:.4f}")
    print(f"[INFO] outputs: {out_dir}")


if __name__ == "__main__":
    main()
