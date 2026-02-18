#!/usr/bin/env python3
"""
Quick regression runner for Stage1-post dirty cases.

Runs postprocess-only from YOLO JSON and asserts key quality conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microfluidics_chip.core.config import TopologyConfig, load_config_from_yaml  # noqa: E402
from microfluidics_chip.pipelines.stage1 import run_stage1_postprocess_from_json  # noqa: E402


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
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


def _resolve_json_path(case_dir: Path) -> Path:
    cand = [
        case_dir / "yolo_raw_detections.json",
        case_dir / "adaptive_yolo_raw_detections.json",
    ]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(f"No detections json found under: {case_dir}")


def _resolve_raw_path(case_dir: Path) -> Path:
    p = case_dir / "raw.png"
    if p.exists():
        return p
    raise FileNotFoundError(f"raw.png not found under: {case_dir}")


def _default_case_dirs() -> List[Path]:
    return [
        REPO_ROOT / "data/processed/chip004/pred/dirty_02",
        REPO_ROOT / "data/processed/chip004/pred/dirty_04",
        REPO_ROOT / "data/processed/chip004/pred/dirty_03",
        REPO_ROOT / "data/processed/chip004/pred/dirty_06",
        REPO_ROOT / "data/processed/chip001/pred/dirty_06",
    ]


def _assert_case(
    case_name: str,
    run_dir: Path,
    metadata: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    qm = metadata.get("quality_metrics", {}) if isinstance(metadata.get("quality_metrics"), dict) else {}
    errors: List[str] = []

    slices_path = run_dir / "chamber_slices.npz"
    if not slices_path.exists():
        errors.append("missing chamber_slices.npz")
    else:
        try:
            payload = np.load(slices_path)
            arr = payload["slices"] if "slices" in payload else payload[payload.files[0]]
            if not (isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[0] == 12):
                errors.append(f"invalid slice shape: {None if not isinstance(arr, np.ndarray) else arr.shape}")
        except Exception as exc:
            errors.append(f"failed to read chamber_slices.npz: {exc}")

    required_keys = [
        "n_det_raw",
        "n_det_dedup",
        "pitch_final",
        "pitch_guard_triggered",
        "fill_ratio",
        "used_real_points",
        "model_chosen",
        "score_sim",
        "score_affine",
        "score_h",
        "reproj_median",
        "reproj_mean",
        "geometry_suspect",
        "slice_mode",
        "blank_selected",
        "blank_is_outermost",
        "blank_valid",
    ]
    for k in required_keys:
        if k not in qm:
            errors.append(f"missing quality_metrics.{k}")

    pitch_final = _safe_float(qm.get("pitch_final"))
    if not np.isfinite(pitch_final) or pitch_final < 20.0:
        errors.append(f"pitch_final<20: {pitch_final}")

    fill_ratio = _safe_float(qm.get("fill_ratio"))
    if "dirty_03" in case_name and np.isfinite(fill_ratio) and fill_ratio > 0.33:
        errors.append(f"dirty_03 fill_ratio>0.33: {fill_ratio:.3f}")

    used_filled_for_transform = _safe_int(qm.get("transform_used_filled_points"), default=0)
    if used_filled_for_transform != 0:
        errors.append(f"transform_used_filled_points!=0: {used_filled_for_transform}")

    blank_selected = qm.get("blank_selected")
    blank_is_outermost = bool(qm.get("blank_is_outermost", False))
    if blank_selected is not None and not blank_is_outermost:
        errors.append(f"blank selected but not outermost: {blank_selected}")

    model_chosen = str(qm.get("model_chosen", ""))
    if model_chosen not in {"similarity", "affine", "homography", "none"}:
        errors.append(f"unexpected model_chosen: {model_chosen}")

    return len(errors) == 0, errors


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage1-post dirty regression and assert quality fields.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/_stage1_post_regression"),
    )
    parser.add_argument(
        "--case-dir",
        dest="case_dirs",
        action="append",
        default=[],
        help="Case directory (e.g. data/processed/chip004/pred/dirty_03). Repeatable.",
    )
    parser.add_argument("--min-topology-detections", type=int, default=8)
    parser.add_argument(
        "--disable-fallback-detection",
        action="store_true",
        help="Disable try2 adaptive fallback detection.",
    )
    parser.add_argument(
        "--strict-semantic",
        action="store_true",
        help="Disable geo export when blank unresolved (for comparison only).",
    )
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config.resolve())
    stage1_cfg = cfg.stage1.model_copy(deep=True)
    if stage1_cfg.topology is None:
        stage1_cfg.topology = TopologyConfig()
    stage1_cfg.topology.success_mode = "geo_only"

    output_root = args.output_root.resolve()
    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    case_dirs = [Path(p).resolve() for p in args.case_dirs] if args.case_dirs else _default_case_dirs()
    case_dirs = [p for p in case_dirs if p.exists()]
    if not case_dirs:
        raise RuntimeError("No valid case directories found.")

    summary_rows: List[Dict[str, Any]] = []
    fail_rows: List[Dict[str, Any]] = []

    for case_dir in case_dirs:
        det_json = _resolve_json_path(case_dir)
        raw_path = _resolve_raw_path(case_dir)
        chip_name = case_dir.parents[1].name if len(case_dir.parents) >= 2 else "chip"
        case_name = case_dir.name
        run_name = f"{chip_name}_{case_name}"
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] running case: {case_dir}")
        run_stage1_postprocess_from_json(
            detections_json_path=det_json,
            output_dir=output_root,
            config=stage1_cfg,
            raw_image_path=raw_path,
            chip_id=run_name,
            min_topology_detections=int(args.min_topology_detections),
            enable_fallback_detection=not bool(args.disable_fallback_detection),
            save_individual_slices=False,
            save_debug=True,
            export_geo_even_if_blank_unresolved=not bool(args.strict_semantic),
        )

        metadata_path = run_dir / "stage1_metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(f"missing stage1_metadata.json: {run_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        qm = metadata.get("quality_metrics", {}) if isinstance(metadata.get("quality_metrics"), dict) else {}

        ok, errors = _assert_case(case_name=case_name, run_dir=run_dir, metadata=metadata)
        row = {
            "case_dir": str(case_dir),
            "run_dir": str(run_dir),
            "ok": int(ok),
            "slice_mode": qm.get("slice_mode"),
            "geometry_suspect": qm.get("geometry_suspect"),
            "pitch_final": qm.get("pitch_final"),
            "fill_ratio": qm.get("fill_ratio"),
            "used_real_points": qm.get("used_real_points"),
            "model_chosen": qm.get("model_chosen"),
            "score_sim": qm.get("score_sim"),
            "score_affine": qm.get("score_affine"),
            "score_h": qm.get("score_h"),
            "reproj_median": qm.get("reproj_median"),
            "reproj_mean": qm.get("reproj_mean"),
            "blank_selected": qm.get("blank_selected"),
            "blank_is_outermost": qm.get("blank_is_outermost"),
            "blank_valid": qm.get("blank_valid"),
            "n_det_raw": qm.get("n_det_raw"),
            "n_det_dedup": qm.get("n_det_dedup"),
        }
        summary_rows.append(row)
        if not ok:
            fail_rows.append(
                {
                    "case_dir": str(case_dir),
                    "run_dir": str(run_dir),
                    "errors": " | ".join(errors),
                }
            )

    _write_csv(output_root / "summary.csv", summary_rows)
    _write_csv(output_root / "assert_failures.csv", fail_rows)
    print(f"[INFO] summary: {output_root / 'summary.csv'}")
    print(f"[INFO] failures: {output_root / 'assert_failures.csv'}")

    if fail_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

