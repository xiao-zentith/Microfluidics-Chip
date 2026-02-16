#!/usr/bin/env python3
"""
Compare Stage1-post BLANK behavior between:
- blank_mode=brightness (legacy)
- blank_mode=chromaticity (new)

Input can be either:
1) raw detection json roots (yolo/adaptive_yolo outputs), or
2) stage1-post roots containing debug_stage1_topology.json.

Outputs:
- summary.csv              (one row per sample per mode)
- compare_summary.csv      (one row per sample, old/new comparison)
- key_cases/              (changed or low-margin cases)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microfluidics_chip.core.config import TopologyConfig, load_config_from_yaml  # noqa: E402
from microfluidics_chip.pipelines.stage1 import run_stage1_postprocess_from_json  # noqa: E402


RAW_DET_JSON_NAMES = ("adaptive_yolo_raw_detections.json", "yolo_raw_detections.json")
TOPO_DEBUG_JSON_NAME = "debug_stage1_topology.json"
MODE_BRIGHTNESS = "brightness"
MODE_CHROMATICITY = "chromaticity"


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


def _safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _discover_input_jsons(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for name in RAW_DET_JSON_NAMES:
        files.extend(input_dir.rglob(name))
    if files:
        return sorted(set(files))
    files = list(input_dir.rglob(TOPO_DEBUG_JSON_NAME))
    return sorted(set(files))


def _normalize_detection_item(det: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(det, dict):
        return None
    center = det.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        cx = _safe_float(center[0], default=float("nan"))
        cy = _safe_float(center[1], default=float("nan"))
    else:
        cx = _safe_float(det.get("x"), default=float("nan"))
        cy = _safe_float(det.get("y"), default=float("nan"))

    bbox = det.get("bbox")
    x = y = w = h = None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x = _safe_int(bbox[0], default=0)
        y = _safe_int(bbox[1], default=0)
        b2 = _safe_int(bbox[2], default=0)
        b3 = _safe_int(bbox[3], default=0)
        # prefer (x,y,w,h); fallback if looks like (x1,y1,x2,y2)
        if b2 > x and b3 > y and (b2 - x) > 4 and (b3 - y) > 4 and (b2 > 3000 or b3 > 3000):
            w = max(1, b2 - x)
            h = max(1, b3 - y)
        else:
            w = max(1, b2)
            h = max(1, b3)

    if not np.isfinite(cx) or not np.isfinite(cy):
        if x is not None and y is not None and w is not None and h is not None:
            cx = float(x + w * 0.5)
            cy = float(y + h * 0.5)
        else:
            return None

    if x is None or y is None or w is None or h is None:
        x = int(round(cx - 1))
        y = int(round(cy - 1))
        w = 2
        h = 2

    conf = det.get("confidence", det.get("conf", 0.0))
    cls = det.get("class_id", 1)
    return {
        "bbox": [int(x), int(y), int(max(1, w)), int(max(1, h))],
        "center": [float(cx), float(cy)],
        "class_id": int(cls),
        "confidence": float(_safe_float(conf, default=0.0)),
    }


def _prepare_detection_json(source_json: Path, temp_input_root: Path) -> Tuple[Path, Dict[str, Any]]:
    payload = _safe_load_json(source_json)
    if not payload:
        raise RuntimeError(f"Invalid json: {source_json}")

    name = source_json.name
    if name in RAW_DET_JSON_NAMES:
        # already compatible
        return source_json, payload

    if name != TOPO_DEBUG_JSON_NAME:
        raise RuntimeError(f"Unsupported input json: {source_json}")

    raw_image_path = payload.get("raw_image_path")
    if not raw_image_path:
        fallback_raw = source_json.parent / "raw.png"
        if fallback_raw.exists():
            raw_image_path = str(fallback_raw)
    if not raw_image_path:
        raise RuntimeError(f"Cannot resolve raw_image_path from {source_json}")

    raw_det = payload.get("detections", [])
    dets: List[Dict[str, Any]] = []
    for det in raw_det:
        norm = _normalize_detection_item(det)
        if norm is not None:
            dets.append(norm)

    if not dets:
        raise RuntimeError(f"No valid detections parsed from {source_json}")

    chip_id = str(payload.get("chip_id") or source_json.parent.name)
    out_payload = {
        "chip_id": chip_id,
        "raw_image_path": str(raw_image_path),
        "roi_bbox": payload.get("roi_bbox"),
        "detections": dets,
        "source_topology_json": str(source_json),
    }
    temp_input_root.mkdir(parents=True, exist_ok=True)
    out_path = temp_input_root / f"{chip_id}.json"
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path, out_payload


def _extract_blank_gt(payload: Dict[str, Any]) -> Optional[int]:
    for key in ("blank_gt", "blank_id_gt", "blank_idx_gt", "blank_id"):
        if key in payload:
            val = _safe_int(payload.get(key), default=-1)
            if 0 <= val < 12:
                return val
    return None


def _run_one_mode(
    mode: str,
    det_json: Path,
    chip_id: str,
    cfg_path: Path,
    template: Optional[Path],
    output_root: Path,
    min_topology_detections: int,
    enable_fallback_detection: bool,
) -> Dict[str, Any]:
    cfg = load_config_from_yaml(cfg_path)
    stage1_cfg = cfg.stage1.model_copy(deep=True)
    if stage1_cfg.topology is None:
        stage1_cfg.topology = TopologyConfig()
    stage1_cfg.topology.blank_mode = str(mode)
    if template is not None:
        stage1_cfg.topology.template_path = str(template)

    run_error = ""
    mode_out = output_root / mode
    mode_out.mkdir(parents=True, exist_ok=True)
    try:
        run_stage1_postprocess_from_json(
            detections_json_path=det_json,
            output_dir=mode_out,
            config=stage1_cfg,
            min_topology_detections=int(min_topology_detections),
            enable_fallback_detection=bool(enable_fallback_detection),
            save_individual_slices=False,
            save_debug=True,
        )
    except Exception as e:
        run_error = str(e)

    run_dir = mode_out / chip_id
    topo = _safe_load_json(run_dir / TOPO_DEBUG_JSON_NAME)
    meta = _safe_load_json(run_dir / "stage1_metadata.json")
    blank = topo.get("blank", {}) if isinstance(topo, dict) else {}
    qc = topo.get("qc", {}) if isinstance(topo, dict) else {}

    blank_id_pred = blank.get("blank_id_pred", blank.get("blank_id"))
    blank_id_pred = None if blank_id_pred is None else _safe_int(blank_id_pred, default=-1)
    if blank_id_pred is not None and blank_id_pred < 0:
        blank_id_pred = None

    blank_score = _safe_float(blank.get("blank_score"), default=float("nan"))
    if not np.isfinite(blank_score):
        scores = blank.get("blank_scores", {})
        if isinstance(scores, dict) and blank_id_pred is not None:
            blank_score = _safe_float(scores.get(str(blank_id_pred), scores.get(blank_id_pred)))

    reference_arm_pred = blank.get("reference_arm_pred", blank.get("reference_arm", qc.get("reference_arm_pred", qc.get("reference_arm"))))
    arm_margin = _safe_float(blank.get("arm_margin", qc.get("arm_margin_pred", qc.get("arm_margin_chromaticity", qc.get("arm_margin_color")))))
    blank_margin = _safe_float(blank.get("blank_margin", qc.get("blank_margin_pred", qc.get("blank_margin_chromaticity", qc.get("blank_margin_color")))))

    blank_status_pred = str(blank.get("blank_status_pred", blank.get("blank_status", qc.get("blank_status_pred", qc.get("blank_status", "unknown")))))
    blank_unresolved = bool(blank.get("blank_unresolved", blank_status_pred == "unresolved"))
    blank_is_outermost = bool(blank.get("blank_is_outermost", blank_id_pred in {2, 5, 8, 11} if blank_id_pred is not None else False))

    final_status = str(topo.get("final_status", topo.get("status", "failed")))
    if run_error and final_status == "success":
        final_status = "failed"

    return {
        "chip_id": chip_id,
        "blank_mode": mode,
        "run_dir": str(run_dir),
        "status": final_status,
        "run_error": run_error,
        "blank_id_pred": blank_id_pred,
        "blank_score": blank_score if np.isfinite(blank_score) else None,
        "reference_arm_pred": None if reference_arm_pred is None else str(reference_arm_pred),
        "arm_margin": arm_margin if np.isfinite(arm_margin) else None,
        "blank_margin": blank_margin if np.isfinite(blank_margin) else None,
        "blank_unresolved": int(blank_unresolved),
        "blank_is_outermost": int(blank_is_outermost),
        "blank_confidence": _safe_float(blank.get("blank_confidence", qc.get("blank_confidence")), default=float("nan")),
        "blank_status_pred": blank_status_pred,
        "blank_status": blank.get("blank_status"),
        "blank_mode_reported": blank.get("blank_mode"),
        "blank_id_old": blank.get("blank_id_old"),
        "blank_id_color": blank.get("blank_id_color"),
        "blank_id_chromaticity": blank.get("blank_id_chromaticity"),
        "quality_gate_passed": meta.get("quality_gate_passed"),
    }


def _copy_case_artifacts(case_dir: Path, run_dir: Path, mode: str) -> None:
    if not run_dir.exists():
        return
    for name in (
        "debug_stage1_topology.png",
        "debug_stage1_topology.json",
        "debug_blank_features.json",
        "debug_overlay.png",
        "stage1_metadata.json",
        "aligned.png",
    ):
        src = run_dir / name
        if src.exists():
            dst = case_dir / f"{mode}_{name}"
            shutil.copy2(src, dst)
    raw = run_dir / "raw.png"
    if raw.exists() and not (case_dir / "raw.png").exists():
        shutil.copy2(raw, case_dir / "raw.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch compare BLANK modes: brightness vs chromaticity")
    parser.add_argument("--input-dir", type=Path, required=True, help="Detection-json root or stage1-post root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for comparison")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--template", type=Path, default=None, help="Optional template json path")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--min-topology-detections", type=int, default=8)
    parser.add_argument("--enable-fallback-detection", action="store_true")
    parser.add_argument("--arm-margin-thr", type=float, default=0.05)
    parser.add_argument("--blank-margin-thr", type=float, default=0.02)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_jsons = _discover_input_jsons(input_dir)
    if not source_jsons:
        raise RuntimeError(f"No supported json found under: {input_dir}")

    source_jsons = source_jsons[: max(1, int(args.max_samples))]
    temp_input_root = output_dir / "_prepared_inputs"
    key_cases_root = output_dir / "key_cases"
    key_cases_root.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []
    changed_cnt = 0

    for idx, src_json in enumerate(source_jsons, start=1):
        prepared_json, prepared_payload = _prepare_detection_json(src_json, temp_input_root)
        chip_id = str(prepared_payload.get("chip_id") or src_json.parent.name)
        blank_gt = _extract_blank_gt(prepared_payload)

        print(f"[{idx}/{len(source_jsons)}] {chip_id}")
        bright = _run_one_mode(
            mode=MODE_BRIGHTNESS,
            det_json=prepared_json,
            chip_id=chip_id,
            cfg_path=args.config,
            template=args.template,
            output_root=output_dir,
            min_topology_detections=args.min_topology_detections,
            enable_fallback_detection=bool(args.enable_fallback_detection),
        )
        chroma = _run_one_mode(
            mode=MODE_CHROMATICITY,
            det_json=prepared_json,
            chip_id=chip_id,
            cfg_path=args.config,
            template=args.template,
            output_root=output_dir,
            min_topology_detections=args.min_topology_detections,
            enable_fallback_detection=bool(args.enable_fallback_detection),
        )

        for rec in (bright, chroma):
            rec_row = {
                "sample_id": chip_id,
                "source_json": str(src_json),
                "prepared_json": str(prepared_json),
                "blank_mode": rec["blank_mode"],
                "reference_arm_pred": rec["reference_arm_pred"],
                "blank_id_pred": rec["blank_id_pred"],
                "arm_margin": rec["arm_margin"],
                "blank_margin": rec["blank_margin"],
                "blank_unresolved": rec["blank_unresolved"],
                "blank_is_outermost": rec["blank_is_outermost"],
                "blank_score": rec["blank_score"],
                "blank_confidence": rec["blank_confidence"],
                "blank_status_pred": rec["blank_status_pred"],
                "status": rec["status"],
                "quality_gate_passed": rec["quality_gate_passed"],
                "run_error": rec["run_error"],
                "run_dir": rec["run_dir"],
                "blank_gt": blank_gt,
                "blank_acc": (
                    int(rec["blank_id_pred"] == blank_gt)
                    if blank_gt is not None and rec["blank_id_pred"] is not None
                    else None
                ),
            }
            run_rows.append(rec_row)

        changed = bright["blank_id_pred"] != chroma["blank_id_pred"]
        consistency = int(not changed)
        chroma_low_margin = (
            (chroma["arm_margin"] is not None and float(chroma["arm_margin"]) < float(args.arm_margin_thr))
            or (chroma["blank_margin"] is not None and float(chroma["blank_margin"]) < float(args.blank_margin_thr))
        )
        compare_row = {
            "sample_id": chip_id,
            "source_json": str(src_json),
            "blank_gt": blank_gt,
            "blank_id_old": bright["blank_id_pred"],
            "blank_id_color": chroma["blank_id_pred"],
            "blank_status_color": chroma["blank_status_pred"],
            "old_status": bright["status"],
            "color_status": chroma["status"],
            "old_unresolved": bright["blank_unresolved"],
            "color_unresolved": chroma["blank_unresolved"],
            "old_arm_margin": bright["arm_margin"],
            "color_arm_margin": chroma["arm_margin"],
            "old_blank_margin": bright["blank_margin"],
            "color_blank_margin": chroma["blank_margin"],
            "blank_id_changed": int(changed),
            "consistency_old_vs_color": consistency,
            "chroma_low_margin_case": int(chroma_low_margin),
            "old_run_dir": bright["run_dir"],
            "color_run_dir": chroma["run_dir"],
        }
        if blank_gt is not None:
            compare_row["old_acc"] = int(bright["blank_id_pred"] == blank_gt) if bright["blank_id_pred"] is not None else None
            compare_row["color_acc"] = int(chroma["blank_id_pred"] == blank_gt) if chroma["blank_id_pred"] is not None else None
        compare_rows.append(compare_row)

        if changed or chroma_low_margin or int(chroma["blank_unresolved"]) == 1:
            changed_cnt += 1
            case_dir = key_cases_root / chip_id
            case_dir.mkdir(parents=True, exist_ok=True)
            _copy_case_artifacts(case_dir, Path(bright["run_dir"]), MODE_BRIGHTNESS)
            _copy_case_artifacts(case_dir, Path(chroma["run_dir"]), MODE_CHROMATICITY)
            # keep source json for traceability
            shutil.copy2(src_json, case_dir / f"source_{src_json.name}")

    if not run_rows:
        raise RuntimeError("No results generated")

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    compare_path = output_dir / "compare_summary.csv"
    with open(compare_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

    old_rows = [r for r in run_rows if r["blank_mode"] == MODE_BRIGHTNESS]
    new_rows = [r for r in run_rows if r["blank_mode"] == MODE_CHROMATICITY]
    old_unresolved_rate = float(np.mean([int(r["blank_unresolved"]) for r in old_rows])) if old_rows else float("nan")
    new_unresolved_rate = float(np.mean([int(r["blank_unresolved"]) for r in new_rows])) if new_rows else float("nan")
    consistency_rate = float(np.mean([int(r["consistency_old_vs_color"]) for r in compare_rows])) if compare_rows else float("nan")

    old_acc_vals = [r.get("blank_acc") for r in old_rows if r.get("blank_acc") is not None]
    new_acc_vals = [r.get("blank_acc") for r in new_rows if r.get("blank_acc") is not None]
    old_acc = float(np.mean(old_acc_vals)) if old_acc_vals else float("nan")
    new_acc = float(np.mean(new_acc_vals)) if new_acc_vals else float("nan")

    print("")
    print(f"Done. summary: {summary_path}")
    print(f"Done. compare: {compare_path}")
    print(f"Samples: {len(source_jsons)}")
    print(f"blank_unresolved_rate old/new: {old_unresolved_rate:.3f} / {new_unresolved_rate:.3f}")
    print(f"blank_id consistency old vs color: {consistency_rate:.3f}")
    if np.isfinite(old_acc) and np.isfinite(new_acc):
        print(f"blank accuracy old/new: {old_acc:.3f} / {new_acc:.3f}")
    else:
        print("blank accuracy old/new: n/a (no blank_gt found)")
    print(f"key cases: {changed_cnt} (saved to {key_cases_root})")


if __name__ == "__main__":
    main()
