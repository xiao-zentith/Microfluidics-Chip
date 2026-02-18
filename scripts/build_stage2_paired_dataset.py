#!/usr/bin/env python3
"""
Build chip-internal paired dataset for Stage2:
- Step A: build GT canonical slices per chip
- Step B: build pred canonical slices for non-ideal samples
- Step C: offline orientation pairing per chip + merge train/val index
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microfluidics_chip.core.config import TopologyConfig, load_config_from_yaml  # noqa: E402
from microfluidics_chip.pipelines.stage1 import (  # noqa: E402
    run_stage1_postprocess_from_json,
    run_stage1_yolo_only,
)
from microfluidics_chip.stage1_detection.detector import ChamberDetector  # noqa: E402


ORDER_R1_U3: Tuple[str, ...] = (
    "R1", "R2", "R3",
    "D1", "D2", "D3",
    "L1", "L2", "L3",
    "U1", "U2", "U3",
)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _classify_fail_bucket(reason: Optional[str]) -> str:
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


def _export_slices_png(npz_path: Path, dst_dir: Path, order: Sequence[str]) -> Tuple[int, Tuple[int, int]]:
    if not npz_path.exists():
        return 0, (0, 0)
    with np.load(npz_path) as data:
        if "slices" in data:
            arr = data["slices"]
        elif len(data.files) > 0:
            arr = data[data.files[0]]
        else:
            return 0, (0, 0)
    arr = np.asarray(arr)
    if arr.ndim != 4:
        return 0, (0, 0)
    n = int(min(arr.shape[0], len(order)))
    h, w = int(arr.shape[1]), int(arr.shape[2])
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        out_path = dst_dir / f"{order[i]}.png"
        import cv2  # local import to reduce startup overhead
        cv2.imwrite(str(out_path), arr[i])
    return n, (h, w)


def _collect_chip_dirs(raw_root: Path) -> List[Path]:
    return sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.lower().startswith("chip")])


def _find_gt_image(chip_dir: Path) -> Optional[Path]:
    for cand in sorted(chip_dir.glob("gt.*")):
        if cand.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return cand
    for cand in sorted(chip_dir.glob("GT.*")):
        if cand.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return cand
    return None


def _find_pred_images(chip_dir: Path, gt_path: Path) -> List[Path]:
    outs: List[Path] = []
    for p in sorted(chip_dir.glob("*.jpg")) + sorted(chip_dir.glob("*.jpeg")):
        if p.name == gt_path.name:
            continue
        if p.name.lower().startswith("debug_"):
            continue
        outs.append(p)
    return outs


def _run_one_stage1_post(
    *,
    chip_id: str,
    sample_id: str,
    image_path: Path,
    output_root: Path,
    stage1_cfg: Any,
    detector: ChamberDetector,
    min_topology_detections: int,
    enable_fallback_detection: bool,
    export_geo_even_if_blank_unresolved: bool,
) -> Dict[str, Any]:
    yolo_out = run_stage1_yolo_only(
        chip_id=sample_id,
        raw_image_path=image_path,
        output_dir=output_root,
        config=stage1_cfg,
        detector=detector,
    )
    det_json = output_root / sample_id / "yolo_raw_detections.json"
    fail_reason = ""
    fail_bucket = ""
    geo_success = False
    semantic_ready = False
    used_fallback = False
    n_det = int(yolo_out.get("num_detections", 0))
    reproj_mean = float("nan")
    reproj_max = float("nan")
    offset_max = float("nan")
    geo_quality_level = ""
    run_dir = output_root / sample_id

    try:
        run_stage1_postprocess_from_json(
            detections_json_path=det_json,
            output_dir=output_root,
            config=stage1_cfg,
            min_topology_detections=int(min_topology_detections),
            enable_fallback_detection=bool(enable_fallback_detection),
            export_geo_even_if_blank_unresolved=bool(export_geo_even_if_blank_unresolved),
            save_individual_slices=False,
            save_debug=True,
        )
    except Exception as e:
        fail_reason = str(e)

    meta = _safe_load_json(run_dir / "stage1_metadata.json")
    topo = _safe_load_json(run_dir / "debug_stage1_topology.json")
    geom = _safe_load_json(run_dir / "debug_geometry_alignment.json")
    qm = meta.get("quality_metrics", {}) if isinstance(meta.get("quality_metrics"), dict) else {}
    qc = topo.get("qc", {}) if isinstance(topo.get("qc"), dict) else {}

    geo_success = bool(meta.get("geo_success", qm.get("geo_success", False)))
    semantic_ready = bool(meta.get("semantic_ready", qm.get("semantic_ready", qm.get("blank_pass", False))))
    used_fallback = bool(meta.get("used_fallback", qm.get("used_fallback", False)))
    n_det = int(
        qc.get(
            "n_det",
            topo.get("n_det", yolo_out.get("num_detections", 0)),
        )
    )
    reproj_mean = _safe_float(meta.get("reprojection_error_mean_px", qm.get("reprojection_error_mean_px", geom.get("reprojection_error_mean_px"))))
    reproj_max = _safe_float(meta.get("reprojection_error_max_px", qm.get("reprojection_error_max_px", geom.get("reprojection_error_max_px"))))
    offset_max = _safe_float(meta.get("slice_center_offset_max_px", qm.get("slice_center_offset_max_px", geom.get("slice_center_offset_max_px"))))
    geo_quality_level = str(meta.get("geo_quality_level", qm.get("geo_quality_level", "")))

    if not geo_success:
        if not fail_reason:
            fail_reason = str(topo.get("final_reason", topo.get("failure_reason", "")))
        fail_bucket = _classify_fail_bucket(fail_reason)

    n_slices, slice_hw = _export_slices_png(
        run_dir / "chamber_slices.npz",
        run_dir / "slices",
        ORDER_R1_U3,
    )

    return {
        "chip_id": chip_id,
        "sample_id": sample_id,
        "image_path": str(image_path),
        "run_dir": str(run_dir),
        "geo_success": int(geo_success),
        "semantic_ready": int(semantic_ready),
        "used_fallback": int(used_fallback),
        "n_det": int(n_det),
        "n_slices": int(n_slices),
        "slice_h": int(slice_hw[0]),
        "slice_w": int(slice_hw[1]),
        "geo_quality_level": geo_quality_level,
        "reprojection_error_mean_px": reproj_mean,
        "reprojection_error_max_px": reproj_max,
        "slice_center_offset_max_px": offset_max,
        "fail_bucket": fail_bucket,
        "fail_reason": fail_reason,
    }


def _copy_failure_case(run_dir: Path, image_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "raw.png",
        "yolo_raw_detections.png",
        "yolo_raw_detections.json",
        "debug_stage1_topology.png",
        "debug_stage1_topology.json",
        "debug_geometry_alignment.json",
        "debug_overlay.png",
    ):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    if image_path.exists():
        shutil.copy2(image_path, dst_dir / image_path.name)


def _summarize_pred_chip(rows: Sequence[Dict[str, Any]], chip_id: str) -> Dict[str, Any]:
    if not rows:
        return {"chip_id": chip_id, "n_samples": 0}
    n = len(rows)
    n_geo = sum(int(r["geo_success"]) for r in rows)
    n_sem = sum(int(r["semantic_ready"]) for r in rows)
    n_fb = sum(int(r["used_fallback"]) for r in rows)
    n_det = np.asarray([int(r["n_det"]) for r in rows], dtype=np.float32)
    fail_counts: Dict[str, int] = {
        "fail_n_det_lt_min": 0,
        "fail_topology_fit": 0,
        "fail_geometry_alignment": 0,
        "fail_io_or_missing_raw": 0,
    }
    for r in rows:
        b = str(r.get("fail_bucket", ""))
        if b in fail_counts:
            fail_counts[b] += 1
    return {
        "chip_id": chip_id,
        "n_samples": int(n),
        "geo_output_rate": float(n_geo / n),
        "semantic_ready_rate": float(n_sem / n),
        "used_fallback_rate": float(n_fb / n),
        "n_det_mean": float(np.mean(n_det)),
        "n_det_median": float(np.median(n_det)),
        "p_n_det_ge_12": float(np.mean(n_det >= 12)),
        "p_n_det_ge_10": float(np.mean(n_det >= 10)),
        "p_n_det_lt_8": float(np.mean(n_det < 8)),
        "fail_n_det_lt_min": int(fail_counts["fail_n_det_lt_min"]),
        "fail_topology_fit": int(fail_counts["fail_topology_fit"]),
        "fail_geometry_alignment": int(fail_counts["fail_geometry_alignment"]),
        "fail_io_or_missing_raw": int(fail_counts["fail_io_or_missing_raw"]),
        "fail_bucket": json.dumps(fail_counts, ensure_ascii=False),
    }


def _merge_pairing_rows(chip_pair_csv: Path, chip_id: str) -> List[Dict[str, Any]]:
    if not chip_pair_csv.exists():
        return []
    out: List[Dict[str, Any]] = []
    with chip_pair_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rec = dict(row)
            rec["chip_id"] = chip_id
            out.append(rec)
    return out


def _load_gt_dir(processed_chip_dir: Path) -> Optional[Path]:
    gt_root = processed_chip_dir / "gt"
    candidates = sorted([p for p in gt_root.iterdir() if p.is_dir()]) if gt_root.exists() else []
    for p in candidates:
        if (p / "chamber_slices.npz").exists():
            return p
    return None


def _build_stage2_indexes(
    *,
    processed_root: Path,
    chip_ids: Sequence[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    for chip_id in chip_ids:
        chip_dir = processed_root / chip_id
        pairing_csv = chip_dir / "pairing" / "paired_index.csv"
        pairing_rows = _merge_pairing_rows(pairing_csv, chip_id=chip_id)
        gt_dir = _load_gt_dir(chip_dir)
        if gt_dir is None:
            continue
        for pr in pairing_rows:
            paired_success = int(float(pr.get("paired_success", 0)))
            if paired_success != 1:
                continue
            sample_id = str(pr.get("sample_id"))
            mapping_raw = pr.get("mapping_pred_to_gt", "[]")
            try:
                mapping = list(json.loads(mapping_raw))
            except Exception:
                continue
            if len(mapping) < 12:
                continue
            pred_slices_dir = chip_dir / "pred" / sample_id / "slices"
            gt_slices_dir = gt_dir / "slices"
            for pred_idx, gt_idx_v in enumerate(mapping[:12]):
                try:
                    gt_idx = int(gt_idx_v)
                except Exception:
                    continue
                if gt_idx < 0 or gt_idx >= 12:
                    continue
                input_slice = pred_slices_dir / f"{ORDER_R1_U3[pred_idx]}.png"
                target_slice = gt_slices_dir / f"{ORDER_R1_U3[gt_idx]}.png"
                if not input_slice.exists() or not target_slice.exists():
                    continue
                all_rows.append(
                    {
                        "chip_id": chip_id,
                        "sample_id": sample_id,
                        "orientation_id": pr.get("chosen_orientation"),
                        "margin": _safe_float(pr.get("margin")),
                        "pred_chamber_id": ORDER_R1_U3[pred_idx],
                        "target_chamber_id": ORDER_R1_U3[gt_idx],
                        "chamber_id": ORDER_R1_U3[gt_idx],
                        "input_slice_path": str(input_slice.resolve()),
                        "target_slice_path": str(target_slice.resolve()),
                    }
                )

    chips_sorted = sorted(set(r["chip_id"] for r in all_rows))
    rng = random.Random(int(seed))
    chips_shuffled = chips_sorted[:]
    rng.shuffle(chips_shuffled)
    n_val = 0
    if chips_shuffled:
        n_val = int(round(len(chips_shuffled) * float(val_ratio)))
        n_val = min(max(1, n_val), len(chips_shuffled) - 1) if len(chips_shuffled) > 1 else 1
    val_chips = set(chips_shuffled[:n_val]) if n_val > 0 else set()

    train_rows = [r for r in all_rows if r["chip_id"] not in val_chips]
    val_rows = [r for r in all_rows if r["chip_id"] in val_chips]
    return all_rows, train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage2 paired dataset from chip-internal raw images.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("configs/templates/pinwheel_v3_centered.json"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--min-topology-detections", type=int, default=8)
    parser.add_argument("--enable-fallback-detection", action="store_true")
    parser.add_argument("--strict-semantic-success", action="store_true")
    parser.add_argument("--pairing-margin-thr", type=float, default=0.02)
    parser.add_argument("--pairing-reference-arm-weight", type=float, default=1.5)
    parser.add_argument("--key-cases-limit", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chips", type=str, default="", help="Comma-separated chip ids (e.g. chip001,chip002); empty means all")
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    processed_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config_from_yaml(args.config.resolve())
    stage1_cfg = cfg.stage1.model_copy(deep=True)
    stage1_cfg.yolo.weights_path = str(args.weights.resolve())
    stage1_cfg.yolo.device = str(args.device)
    stage1_cfg.yolo.confidence_threshold = float(args.conf)
    if stage1_cfg.topology is None:
        stage1_cfg.topology = TopologyConfig()
    stage1_cfg.topology.template_path = str(args.template.resolve())
    stage1_cfg.topology.success_mode = "geo_and_semantic" if bool(args.strict_semantic_success) else "geo_only"

    detector = ChamberDetector(stage1_cfg.yolo)

    chips = _collect_chip_dirs(raw_root)
    if args.chips.strip():
        keep = {x.strip() for x in args.chips.split(",") if x.strip()}
        chips = [p for p in chips if p.name in keep]
    if not chips:
        raise RuntimeError(f"No chip dirs found under {raw_root}")

    print(f"[INFO] chips={len(chips)} raw_root={raw_root}")
    print(f"[INFO] processed_root={processed_root}")
    print(
        f"[INFO] stage1 conf={args.conf:.3f}, min_topology_detections={args.min_topology_detections}, "
        f"success_mode={stage1_cfg.topology.success_mode}, enable_fallback_detection={bool(args.enable_fallback_detection)}"
    )

    gt_rows: List[Dict[str, Any]] = []
    pred_detail_rows: List[Dict[str, Any]] = []
    pred_chip_rows: List[Dict[str, Any]] = []
    pairing_chip_rows: List[Dict[str, Any]] = []

    # Step A + B
    for chip_dir in chips:
        chip_id = chip_dir.name
        out_chip = processed_root / chip_id
        out_gt_root = out_chip / "gt"
        out_pred_root = out_chip / "pred"
        out_key_root = out_chip / "key_cases"
        out_gt_root.mkdir(parents=True, exist_ok=True)
        out_pred_root.mkdir(parents=True, exist_ok=True)
        out_key_root.mkdir(parents=True, exist_ok=True)

        gt_img = _find_gt_image(chip_dir)
        if gt_img is None:
            gt_rows.append(
                {
                    "chip_id": chip_id,
                    "gt_source": "",
                    "geo_success": 0,
                    "reprojection_error_mean_px": float("nan"),
                    "slice_center_offset_max_px": float("nan"),
                    "fail_bucket": "fail_io_or_missing_raw",
                    "fail_reason": "gt_not_found",
                }
            )
            continue

        gt_sample_id = "gt"
        gt_rec = _run_one_stage1_post(
            chip_id=chip_id,
            sample_id=gt_sample_id,
            image_path=gt_img,
            output_root=out_gt_root,
            stage1_cfg=stage1_cfg,
            detector=detector,
            min_topology_detections=int(args.min_topology_detections),
            enable_fallback_detection=bool(args.enable_fallback_detection),
            export_geo_even_if_blank_unresolved=(not bool(args.strict_semantic_success)),
        )
        gt_rows.append(
            {
                "chip_id": chip_id,
                "gt_source": str(gt_img),
                "geo_success": int(gt_rec["geo_success"]),
                "reprojection_error_mean_px": gt_rec["reprojection_error_mean_px"],
                "slice_center_offset_max_px": gt_rec["slice_center_offset_max_px"],
                "fail_bucket": gt_rec["fail_bucket"],
                "fail_reason": gt_rec["fail_reason"],
            }
        )
        gt_meta_path = out_gt_root / gt_sample_id / "gt_meta.json"
        gt_meta_payload = {
            "chip_id": chip_id,
            "gt_source": str(gt_img),
            "template_version": Path(args.template).name,
            "slice_size": [int(gt_rec["slice_h"]), int(gt_rec["slice_w"])],
            "order": list(ORDER_R1_U3),
        }
        gt_meta_path.write_text(json.dumps(gt_meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        pred_imgs = _find_pred_images(chip_dir, gt_img)
        chip_pred_rows: List[Dict[str, Any]] = []
        for img_path in pred_imgs:
            sample_id = img_path.stem
            rec = _run_one_stage1_post(
                chip_id=chip_id,
                sample_id=sample_id,
                image_path=img_path,
                output_root=out_pred_root,
                stage1_cfg=stage1_cfg,
                detector=detector,
                min_topology_detections=int(args.min_topology_detections),
                enable_fallback_detection=bool(args.enable_fallback_detection),
                export_geo_even_if_blank_unresolved=(not bool(args.strict_semantic_success)),
            )
            chip_pred_rows.append(rec)
            pred_detail_rows.append(rec)
            if int(rec["geo_success"]) != 1:
                fail_bucket = str(rec.get("fail_bucket", "fail_io_or_missing_raw")) or "fail_io_or_missing_raw"
                dst = out_key_root / "pred_fail" / fail_bucket / sample_id
                _copy_failure_case(Path(rec["run_dir"]), img_path, dst)

        chip_summary = _summarize_pred_chip(chip_pred_rows, chip_id=chip_id)
        pred_chip_rows.append(chip_summary)

        # Step C: pairing per chip (only if GT geo success and pred exists)
        pair_out = out_chip / "pairing"
        pair_out.mkdir(parents=True, exist_ok=True)
        if int(gt_rec["geo_success"]) == 1 and len(chip_pred_rows) > 0:
            cmd = [
                sys.executable,
                str((REPO_ROOT / "scripts" / "offline_orientation_pairing.py").resolve()),
                "--pred-root", str(out_pred_root),
                "--gt-root", str(out_gt_root),
                "--output-dir", str(pair_out),
                "--single-gt",
                "--margin-thr", str(float(args.pairing_margin_thr)),
                "--reference-arm-weight", str(float(args.pairing_reference_arm_weight)),
                "--key-cases-limit", str(int(args.key_cases_limit)),
            ]
            subprocess.run(cmd, check=True)
            pair_summary = _safe_load_json(pair_out / "pairing_summary.json")
            pairing_chip_rows.append(
                {
                    "chip_id": chip_id,
                    "paired_success_rate": pair_summary.get("paired_success_rate"),
                    "margin_p50": pair_summary.get("margin_p50"),
                    "margin_p90": pair_summary.get("margin_p90"),
                    "n_total": pair_summary.get("n_total"),
                    "n_success": pair_summary.get("n_success"),
                    "fail_bucket_low_margin": pair_summary.get("fail_bucket_low_margin"),
                    "fail_bucket_gt_not_found": pair_summary.get("fail_bucket_gt_not_found"),
                    "fail_bucket_npz_invalid_or_missing": pair_summary.get("fail_bucket_npz_invalid_or_missing"),
                }
            )
        else:
            pairing_chip_rows.append(
                {
                    "chip_id": chip_id,
                    "paired_success_rate": float("nan"),
                    "margin_p50": float("nan"),
                    "margin_p90": float("nan"),
                    "n_total": 0,
                    "n_success": 0,
                    "fail_bucket_low_margin": 0,
                    "fail_bucket_gt_not_found": 0,
                    "fail_bucket_npz_invalid_or_missing": 0,
                }
            )

    # Write step summaries
    gt_csv = processed_root / "gt_build_summary.csv"
    pred_csv = processed_root / "pred_build_summary.csv"
    pred_detail_csv = processed_root / "pred_build_details.csv"
    pairing_csv = processed_root / "pairing_summary.csv"
    _write_csv(gt_csv, gt_rows)
    _write_csv(pred_csv, pred_chip_rows)
    _write_csv(pred_detail_csv, pred_detail_rows)
    _write_csv(pairing_csv, pairing_chip_rows)

    # Merge indexes
    chip_ids = [p.name for p in chips]
    all_rows, train_rows, val_rows = _build_stage2_indexes(
        processed_root=processed_root,
        chip_ids=chip_ids,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    stage2_all_csv = processed_root / "stage2_all_index.csv"
    stage2_train_csv = processed_root / "stage2_train_index.csv"
    stage2_val_csv = processed_root / "stage2_val_index.csv"
    _write_csv(stage2_all_csv, all_rows)
    _write_csv(stage2_train_csv, train_rows)
    _write_csv(stage2_val_csv, val_rows)

    # Global summary
    n_gt = len(gt_rows)
    n_gt_ok = sum(int(r.get("geo_success", 0)) for r in gt_rows)
    n_pred = len(pred_detail_rows)
    n_pred_geo = sum(int(r.get("geo_success", 0)) for r in pred_detail_rows)
    pair_rates = [float(r.get("paired_success_rate")) for r in pairing_chip_rows if np.isfinite(_safe_float(r.get("paired_success_rate")))]
    global_summary = {
        "n_chips": len(chips),
        "gt_geo_success_count": int(n_gt_ok),
        "gt_geo_success_rate": float(n_gt_ok / n_gt) if n_gt > 0 else float("nan"),
        "pred_total_samples": int(n_pred),
        "pred_geo_output_count": int(n_pred_geo),
        "pred_geo_output_rate": float(n_pred_geo / n_pred) if n_pred > 0 else float("nan"),
        "pairing_mean_success_rate_per_chip": float(np.mean(pair_rates)) if pair_rates else float("nan"),
        "stage2_all_pairs": int(len(all_rows)),
        "stage2_train_pairs": int(len(train_rows)),
        "stage2_val_pairs": int(len(val_rows)),
        "weights": str(args.weights),
        "template": str(args.template),
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
    }
    (processed_root / "build_summary.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[INFO] build complete")
    print(f"[INFO] gt summary: {gt_csv}")
    print(f"[INFO] pred summary: {pred_csv}")
    print(f"[INFO] pred details: {pred_detail_csv}")
    print(f"[INFO] pairing summary: {pairing_csv}")
    print(f"[INFO] stage2 train index: {stage2_train_csv}")
    print(f"[INFO] stage2 val index: {stage2_val_csv}")
    print(f"[INFO] global summary: {processed_root / 'build_summary.json'}")
    print(
        "[INFO] metrics: "
        f"gt_geo={global_summary['gt_geo_success_count']}/{global_summary['n_chips']} "
        f"pred_geo_rate={global_summary['pred_geo_output_rate']:.4f} "
        f"stage2_pairs={global_summary['stage2_all_pairs']}"
    )


if __name__ == "__main__":
    # Keep OpenMP runtime tolerant in mixed envs
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()

