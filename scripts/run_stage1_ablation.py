#!/usr/bin/env python3
"""
Stage1 dual-class ablation (detection-only + pipeline).

Outputs:
- detection_ablation_table.csv
- pipeline_ablation_table.csv
- ablation_report.md
- key_cases/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from microfluidics_chip.core.config import TopologyConfig, load_config_from_yaml  # noqa: E402
from microfluidics_chip.pipelines.stage1 import run_stage1_postprocess_batch  # noqa: E402
from microfluidics_chip.stage1_detection.topology_fitter import (  # noqa: E402
    TopologyConfig as InternalTopologyConfig,
    TopologyFitter,
)


@dataclass
class BoxRecord:
    cls: int
    conf: float
    bbox_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]


@dataclass
class ImageSample:
    image_path: Path
    label_path: Path
    chip_id: str
    width: int
    height: int
    gt_boxes: List[BoxRecord]
    gt_blank_template_idx: Optional[int] = None
    gt_reference_arm: Optional[str] = None


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


def _parse_conf_list(conf_list_text: str) -> List[float]:
    values: List[float] = []
    for token in conf_list_text.split(","):
        token = token.strip()
        if not token:
            continue
        c = float(token)
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"Invalid conf value: {token}")
        values.append(float(c))
    if not values:
        raise ValueError("conf list is empty")
    return sorted(set(values))


def _resolve_dataset_paths(data_yaml: Path, split: str) -> Tuple[List[Path], Path, Dict[int, str]]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid data yaml: {data_yaml}")

    split_value = payload.get(split)
    if not isinstance(split_value, str) or not split_value:
        raise RuntimeError(f"Split '{split}' missing in {data_yaml}")

    images_dir = Path(split_value)
    if not images_dir.is_absolute():
        images_dir = (data_yaml.parent / images_dir).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    if "images" in images_dir.parts:
        idx = images_dir.parts.index("images")
        labels_dir = Path(*images_dir.parts[:idx], "labels", *images_dir.parts[idx + 1 :])
    else:
        labels_dir = images_dir.parent / "labels" / images_dir.name
    labels_dir = labels_dir.resolve()
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in image_exts])
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    raw_names = payload.get("names", {})
    names: Dict[int, str] = {}
    if isinstance(raw_names, dict):
        for k, v in raw_names.items():
            idx = _safe_int(k, default=-1)
            if idx >= 0:
                names[idx] = str(v)
    elif isinstance(raw_names, list):
        for i, v in enumerate(raw_names):
            names[int(i)] = str(v)
    return image_paths, labels_dir, names


def _parse_yolo_label_line(parts: List[float], width: int, height: int) -> Optional[BoxRecord]:
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    values = parts[1:]

    if len(values) == 4:
        cx_n, cy_n, w_n, h_n = values
        cx = float(cx_n * width)
        cy = float(cy_n * height)
        bw = float(max(1.0, w_n * width))
        bh = float(max(1.0, h_n * height))
        x1 = cx - bw * 0.5
        y1 = cy - bh * 0.5
        x2 = cx + bw * 0.5
        y2 = cy + bh * 0.5
    else:
        if len(values) < 6 or len(values) % 2 != 0:
            return None
        xs = np.asarray(values[0::2], dtype=np.float32) * float(width)
        ys = np.asarray(values[1::2], dtype=np.float32) * float(height)
        x1, x2 = float(np.min(xs)), float(np.max(xs))
        y1, y2 = float(np.min(ys)), float(np.max(ys))
        cx = float((x1 + x2) * 0.5)
        cy = float((y1 + y2) * 0.5)

    x1 = float(max(0.0, min(float(width - 1), x1)))
    y1 = float(max(0.0, min(float(height - 1), y1)))
    x2 = float(max(0.0, min(float(width - 1), x2)))
    y2 = float(max(0.0, min(float(height - 1), y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    return BoxRecord(
        cls=int(cls),
        conf=1.0,
        bbox_xyxy=(x1, y1, x2, y2),
        center_xy=(cx, cy),
    )


def _load_gt_samples(image_paths: List[Path], labels_dir: Path) -> List[ImageSample]:
    out: List[ImageSample] = []
    for image_path in image_paths:
        chip_id = image_path.stem
        label_path = labels_dir / f"{chip_id}.txt"

        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]

        gt_boxes: List[BoxRecord] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s:
                    continue
                parts = [float(x) for x in s.split()]
                item = _parse_yolo_label_line(parts, width=w, height=h)
                if item is not None:
                    gt_boxes.append(item)

        out.append(
            ImageSample(
                image_path=image_path,
                label_path=label_path,
                chip_id=chip_id,
                width=int(w),
                height=int(h),
                gt_boxes=gt_boxes,
            )
        )
    return out


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _match_by_class_iou(
    preds: Sequence[BoxRecord],
    gts: Sequence[BoxRecord],
    iou_thr: float = 0.5,
) -> Dict[str, Any]:
    matched_gt: Dict[int, set] = {}
    tp: Dict[int, int] = {}
    fp: Dict[int, int] = {}
    fn: Dict[int, int] = {}

    classes = sorted(set([int(x.cls) for x in preds] + [int(x.cls) for x in gts]))
    for cls in classes:
        cls_pred_idx = [i for i, p in enumerate(preds) if int(p.cls) == int(cls)]
        cls_gt_idx = [i for i, g in enumerate(gts) if int(g.cls) == int(cls)]
        used = set()
        tp_cls = 0
        for pi in sorted(cls_pred_idx, key=lambda k: float(preds[k].conf), reverse=True):
            best_iou = 0.0
            best_g = -1
            for gi in cls_gt_idx:
                if gi in used:
                    continue
                iou = _bbox_iou(preds[pi].bbox_xyxy, gts[gi].bbox_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_g = gi
            if best_g >= 0 and best_iou >= float(iou_thr):
                used.add(best_g)
                tp_cls += 1
        matched_gt[int(cls)] = used
        tp[int(cls)] = int(tp_cls)
        fp[int(cls)] = int(max(0, len(cls_pred_idx) - tp_cls))
        fn[int(cls)] = int(max(0, len(cls_gt_idx) - tp_cls))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_gt_indices_by_class": matched_gt,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float32)
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _median(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.median(vals))


def _safe_rate(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def _draw_detection_overlay(
    image_path: Path,
    gt_boxes: Sequence[BoxRecord],
    pred_boxes: Sequence[BoxRecord],
    title_lines: Sequence[str],
    out_path: Path,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return

    for g in gt_boxes:
        x1, y1, x2, y2 = g.bbox_xyxy
        color = (255, 200, 0) if int(g.cls) == 0 else (255, 120, 0)
        cv2.rectangle(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, 2)
        cv2.putText(
            img,
            f"GT cls={int(g.cls)}",
            (int(round(x1)), max(18, int(round(y1)) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    for p in pred_boxes:
        x1, y1, x2, y2 = p.bbox_xyxy
        color = (0, 255, 0) if int(p.cls) == 0 else (0, 180, 255)
        cv2.rectangle(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, 2)
        cv2.putText(
            img,
            f"Pred cls={int(p.cls)} conf={float(p.conf):.3f}",
            (int(round(x1)), min(img.shape[0] - 8, int(round(y2)) + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    lines = list(title_lines)
    x0 = 10
    y0 = 24
    max_w = 0
    for ln in lines:
        (w, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        max_w = max(max_w, w)
    cv2.rectangle(img, (x0 - 8, y0 - 18), (x0 + max_w + 10, y0 + 24 * len(lines) + 2), (0, 0, 0), -1)
    cv2.rectangle(img, (x0 - 8, y0 - 18), (x0 + max_w + 10, y0 + 24 * len(lines) + 2), (80, 80, 80), 1)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (x0, y0 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def _conf_tag(conf: float) -> str:
    s = f"{conf:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _run_val_per_class_metrics(
    model: YOLO,
    data_yaml: Path,
    split: str,
    imgsz: int,
    nms_iou: float,
    max_det: int,
    device: str,
) -> Dict[str, Any]:
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=int(imgsz),
        conf=0.001,
        iou=float(nms_iou),
        max_det=int(max_det),
        device=device,
        verbose=False,
        plots=False,
        save=False,
        save_json=False,
    )
    box = metrics.box
    maps = np.asarray(getattr(box, "maps", []), dtype=np.float32)
    p = np.asarray(getattr(box, "p", []), dtype=np.float32)
    r = np.asarray(getattr(box, "r", []), dtype=np.float32)
    all_ap = np.asarray(getattr(box, "all_ap", []), dtype=np.float32)
    names = metrics.names if isinstance(metrics.names, dict) else {}

    per_class: Dict[str, Dict[str, float]] = {}
    n_cls = int(max(len(maps), len(p), len(r), all_ap.shape[0] if all_ap.ndim == 2 else 0))
    for cls in range(n_cls):
        name = str(names.get(cls, f"class_{cls}"))
        ap50 = float(all_ap[cls, 0]) if all_ap.ndim == 2 and cls < all_ap.shape[0] else float("nan")
        ap5095 = float(maps[cls]) if cls < len(maps) else float("nan")
        precision = float(p[cls]) if cls < len(p) else float("nan")
        recall = float(r[cls]) if cls < len(r) else float("nan")
        per_class[name] = {
            "class_id": int(cls),
            "precision": precision,
            "recall": recall,
            "ap50": ap50,
            "ap50_95": ap5095,
        }

    return {
        "overall": {
            "map50": float(getattr(box, "map50", float("nan"))),
            "map50_95": float(getattr(box, "map", float("nan"))),
        },
        "per_class": per_class,
    }


def _predict_all_images(
    model: YOLO,
    samples: Sequence[ImageSample],
    min_conf: float,
    imgsz: int,
    nms_iou: float,
    max_det: int,
    device: str,
) -> Dict[str, List[BoxRecord]]:
    source_paths = [str(x.image_path) for x in samples]
    preds: Dict[str, List[BoxRecord]] = {}
    for result in model.predict(
        source=source_paths,
        conf=float(min_conf),
        iou=float(nms_iou),
        imgsz=int(imgsz),
        max_det=int(max_det),
        device=device,
        verbose=False,
        stream=True,
    ):
        image_path = Path(str(result.path))
        chip_id = image_path.stem
        items: List[BoxRecord] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(np.int32)
            conf = result.boxes.conf.cpu().numpy()
            xywh = result.boxes.xywh.cpu().numpy()
            for i in range(len(xyxy)):
                items.append(
                    BoxRecord(
                        cls=int(cls[i]),
                        conf=float(conf[i]),
                        bbox_xyxy=(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])),
                        center_xy=(float(xywh[i][0]), float(xywh[i][1])),
                    )
                )
        preds[chip_id] = items
    return preds


def _filter_preds_by_conf(preds: Sequence[BoxRecord], conf_thr: float) -> List[BoxRecord]:
    return [p for p in preds if float(p.conf) >= float(conf_thr)]


def _write_detection_json(
    sample: ImageSample,
    preds: Sequence[BoxRecord],
    out_json: Path,
) -> None:
    dets: List[Dict[str, Any]] = []
    for p in preds:
        x1, y1, x2, y2 = p.bbox_xyxy
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        dets.append(
            {
                "bbox": [int(round(x1)), int(round(y1)), int(round(w)), int(round(h))],
                "center": [float(p.center_xy[0]), float(p.center_xy[1])],
                "class_id": int(p.cls),
                "confidence": float(p.conf),
            }
        )
    payload = {
        "chip_id": sample.chip_id,
        "raw_image_path": str(sample.image_path.resolve()),
        "num_detections": len(dets),
        "detections": dets,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _arm_name_from_template_id(template_id: str) -> Optional[str]:
    t = str(template_id).strip().upper()
    if t.startswith("U"):
        return "Up"
    if t.startswith("R"):
        return "Right"
    if t.startswith("D"):
        return "Down"
    if t.startswith("L"):
        return "Left"
    return None


def _load_template_ids(template_path: Optional[Path]) -> List[str]:
    if template_path is None or not template_path.exists():
        return [str(i) for i in range(12)]
    try:
        payload = json.loads(template_path.read_text(encoding="utf-8"))
        geometry = payload.get("geometry", {}) if isinstance(payload, dict) else {}
        points = geometry.get("points", []) if isinstance(geometry, dict) else []
        ids: List[str] = []
        for pt in points:
            if isinstance(pt, dict) and "id" in pt:
                ids.append(str(pt["id"]))
        if len(ids) >= 12:
            return ids[:12]
    except Exception:
        pass
    return [str(i) for i in range(12)]


def _linear_sum_assignment_fallback(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = cost.shape
    pairs: List[Tuple[float, int, int]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            pairs.append((float(cost[r, c]), int(r), int(c)))
    pairs.sort(key=lambda x: x[0])
    used_r = set()
    used_c = set()
    rs: List[int] = []
    cs: List[int] = []
    for _, r, c in pairs:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        rs.append(r)
        cs.append(c)
        if len(rs) >= min(n_rows, n_cols):
            break
    return np.asarray(rs, dtype=np.int32), np.asarray(cs, dtype=np.int32)


def _assign_gt_blank_template(
    sample: ImageSample,
    template_ids: Sequence[str],
    fitter: TopologyFitter,
) -> Tuple[Optional[int], Optional[str]]:
    gt_centers = np.asarray([list(b.center_xy) for b in sample.gt_boxes], dtype=np.float32)
    gt_classes = [int(b.cls) for b in sample.gt_boxes]
    if gt_centers.shape[0] < 4:
        return None, None
    blank_gt_indices = [i for i, c in enumerate(gt_classes) if int(c) == 0]
    if len(blank_gt_indices) != 1:
        return None, None

    fit = fitter.fit_and_fill(
        detected_centers=gt_centers,
        image_shape=(int(sample.height), int(sample.width)),
        image=None,
    )
    if fit.fitted_centers.shape[0] < 12:
        return None, None

    fitted = np.asarray(fit.fitted_centers[:12], dtype=np.float32)
    cost = np.linalg.norm(fitted[:, None, :] - gt_centers[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, cols = linear_sum_assignment(cost)
    except Exception:
        rows, cols = _linear_sum_assignment_fallback(cost)

    gt_to_template: Dict[int, int] = {}
    for r, c in zip(rows.tolist(), cols.tolist()):
        gt_to_template[int(c)] = int(r)

    blank_gt_i = int(blank_gt_indices[0])
    blank_tpl = gt_to_template.get(blank_gt_i)
    if blank_tpl is None:
        return None, None
    arm_name = _arm_name_from_template_id(str(template_ids[blank_tpl]))
    return int(blank_tpl), arm_name


def _prepare_gt_template_mapping(
    samples: Sequence[ImageSample],
    stage1_topology: TopologyConfig,
    template_path: Optional[Path],
) -> None:
    template_ids = _load_template_ids(template_path)
    internal_cfg = InternalTopologyConfig(
        template_scale=float(stage1_topology.template_scale),
        template_path=str(template_path) if template_path is not None else None,
        ransac_iters=int(stage1_topology.ransac_iters),
        ransac_threshold=float(stage1_topology.ransac_threshold),
        min_inliers=int(stage1_topology.min_inliers),
        visibility_margin=int(stage1_topology.visibility_margin),
        brightness_roi_size=int(stage1_topology.brightness_roi_size),
        dark_percentile=float(stage1_topology.dark_percentile),
        fallback_to_affine=bool(stage1_topology.fallback_to_affine),
    )
    fitter = TopologyFitter(internal_cfg)
    for s in samples:
        blank_tpl, arm = _assign_gt_blank_template(s, template_ids=template_ids, fitter=fitter)
        s.gt_blank_template_idx = blank_tpl
        s.gt_reference_arm = arm


def _evaluate_detection_conf(
    *,
    conf: float,
    samples: Sequence[ImageSample],
    all_preds: Dict[str, List[BoxRecord]],
    val_metrics: Dict[str, Any],
    det_json_root: Path,
    key_cases_root: Path,
    key_cases_limit: int,
) -> Tuple[Dict[str, Any], Dict[str, List[str]], Dict[str, List[BoxRecord]]]:
    n_total = len(samples)
    n_det_total: List[int] = []
    n_det_blank: List[int] = []
    n_det_lit: List[int] = []

    tp_total: Dict[int, int] = {0: 0, 1: 0}
    fp_total: Dict[int, int] = {0: 0, 1: 0}
    fn_total: Dict[int, int] = {0: 0, 1: 0}
    gt_total: Dict[int, int] = {0: 0, 1: 0}

    blank_detected_any = 0
    blank_detected_matched = 0
    blank_miss_lit_full_cases: List[str] = []
    over_detection_cases: List[str] = []

    filtered_preds_map: Dict[str, List[BoxRecord]] = {}

    for sample in samples:
        raw_preds = all_preds.get(sample.chip_id, [])
        preds = _filter_preds_by_conf(raw_preds, conf_thr=conf)
        filtered_preds_map[sample.chip_id] = preds

        _write_detection_json(
            sample=sample,
            preds=preds,
            out_json=det_json_root / sample.chip_id / "yolo_raw_detections.json",
        )

        n_total_det = len(preds)
        n_blank_det = sum(1 for p in preds if int(p.cls) == 0)
        n_lit_det = sum(1 for p in preds if int(p.cls) == 1)
        n_det_total.append(int(n_total_det))
        n_det_blank.append(int(n_blank_det))
        n_det_lit.append(int(n_lit_det))
        if n_blank_det > 0:
            blank_detected_any += 1

        match = _match_by_class_iou(preds, sample.gt_boxes, iou_thr=0.5)
        for cls in [0, 1]:
            tp_total[cls] += int(match["tp"].get(cls, 0))
            fp_total[cls] += int(match["fp"].get(cls, 0))
            fn_total[cls] += int(match["fn"].get(cls, 0))
            gt_total[cls] += int(sum(1 for g in sample.gt_boxes if int(g.cls) == cls))

        blank_gt = int(sum(1 for g in sample.gt_boxes if int(g.cls) == 0))
        lit_gt = int(sum(1 for g in sample.gt_boxes if int(g.cls) == 1))
        blank_tp = int(match["tp"].get(0, 0))
        lit_tp = int(match["tp"].get(1, 0))
        if blank_gt > 0 and blank_tp > 0:
            blank_detected_matched += 1
        if blank_gt > 0 and blank_tp == 0 and lit_gt > 0 and lit_tp == lit_gt:
            blank_miss_lit_full_cases.append(sample.chip_id)
        if n_total_det >= 14:
            over_detection_cases.append(sample.chip_id)

    cls_name_map: Dict[int, str] = {}
    per_class_payload = val_metrics.get("per_class", {})
    for name, rec in per_class_payload.items():
        cls_id = _safe_int(rec.get("class_id"), default=-1)
        if cls_id >= 0:
            cls_name_map[cls_id] = str(name)

    # detection key cases overlays
    kept_blank_cases = blank_miss_lit_full_cases[: max(0, key_cases_limit)]
    for chip_id in kept_blank_cases:
        sample = next((s for s in samples if s.chip_id == chip_id), None)
        if sample is None:
            continue
        preds = filtered_preds_map.get(chip_id, [])
        out_path = key_cases_root / "detection_blank_miss_lit_full" / f"conf_{_conf_tag(conf)}" / f"{chip_id}.png"
        _draw_detection_overlay(
            image_path=sample.image_path,
            gt_boxes=sample.gt_boxes,
            pred_boxes=preds,
            title_lines=[
                f"chip={chip_id} conf={conf:.3f}",
                "case=BLANK missed but LIT fully detected",
                f"n_det_total={len(preds)} n_det_blank={sum(1 for p in preds if p.cls==0)}",
            ],
            out_path=out_path,
        )
    kept_overdet_cases = over_detection_cases[: max(0, key_cases_limit)]
    for chip_id in kept_overdet_cases:
        sample = next((s for s in samples if s.chip_id == chip_id), None)
        if sample is None:
            continue
        preds = filtered_preds_map.get(chip_id, [])
        out_path = key_cases_root / "detection_over_detection" / f"conf_{_conf_tag(conf)}" / f"{chip_id}.png"
        _draw_detection_overlay(
            image_path=sample.image_path,
            gt_boxes=sample.gt_boxes,
            pred_boxes=preds,
            title_lines=[
                f"chip={chip_id} conf={conf:.3f}",
                "case=possible false positives / over-detection",
                f"n_det_total={len(preds)}",
            ],
            out_path=out_path,
        )

    recall_blank = _safe_rate(tp_total[0], gt_total[0])
    recall_lit = _safe_rate(tp_total[1], gt_total[1])
    precision_blank = _safe_rate(tp_total[0], tp_total[0] + fp_total[0])
    precision_lit = _safe_rate(tp_total[1], tp_total[1] + fp_total[1])

    det_row: Dict[str, Any] = {
        "conf": float(conf),
        "n_images": int(n_total),
        "mean_n_det_total": _mean(n_det_total),
        "median_n_det_total": _median(n_det_total),
        "p05_n_det_total": _quantile(n_det_total, 5),
        "p95_n_det_total": _quantile(n_det_total, 95),
        "mean_n_det_blank": _mean(n_det_blank),
        "median_n_det_blank": _median(n_det_blank),
        "p05_n_det_blank": _quantile(n_det_blank, 5),
        "p95_n_det_blank": _quantile(n_det_blank, 95),
        "mean_n_det_lit": _mean(n_det_lit),
        "median_n_det_lit": _median(n_det_lit),
        "p05_n_det_lit": _quantile(n_det_lit, 5),
        "p95_n_det_lit": _quantile(n_det_lit, 95),
        "p_n_det_total_ge_12": _safe_rate(sum(1 for x in n_det_total if x >= 12), n_total),
        "p_n_det_total_ge_10": _safe_rate(sum(1 for x in n_det_total if x >= 10), n_total),
        "p_n_det_total_lt_8": _safe_rate(sum(1 for x in n_det_total if x < 8), n_total),
        "p_blank_detected_any": _safe_rate(blank_detected_any, n_total),
        "p_blank_detected_matched": _safe_rate(blank_detected_matched, n_total),
        "blank_recall_iou50_at_conf": recall_blank,
        "lit_recall_iou50_at_conf": recall_lit,
        "blank_precision_iou50_at_conf": precision_blank,
        "lit_precision_iou50_at_conf": precision_lit,
        "blank_miss_lit_full_cases": int(len(blank_miss_lit_full_cases)),
        "over_detection_cases": int(len(over_detection_cases)),
    }

    # add per-class AP/Recall from val
    for cls_id in [0, 1]:
        cls_name = cls_name_map.get(cls_id, f"class_{cls_id}")
        rec = per_class_payload.get(cls_name, {})
        det_row[f"{cls_name}_ap50"] = _safe_float(rec.get("ap50"))
        det_row[f"{cls_name}_ap50_95"] = _safe_float(rec.get("ap50_95"))
        det_row[f"{cls_name}_recall_val"] = _safe_float(rec.get("recall"))
        det_row[f"{cls_name}_precision_val"] = _safe_float(rec.get("precision"))

    case_lists = {
        "blank_miss_lit_full": blank_miss_lit_full_cases,
        "over_detection": over_detection_cases,
    }
    return det_row, case_lists, filtered_preds_map


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_top2_margin(items: Any) -> float:
    if not isinstance(items, list) or len(items) < 2:
        return float("nan")
    s1 = _safe_float((items[0] or {}).get("score"))
    s2 = _safe_float((items[1] or {}).get("score"))
    if not np.isfinite(s1) or not np.isfinite(s2):
        return float("nan")
    return float(s2 - s1)


def _evaluate_pipeline_conf(
    *,
    conf: float,
    samples: Sequence[ImageSample],
    detections_root: Path,
    pipeline_root: Path,
    stage1_cfg: Any,
    min_topology_detections: int,
    enable_fallback_detection: bool,
    key_cases_root: Path,
    key_cases_limit: int,
) -> Dict[str, Any]:
    # run postprocess batch
    run_stage1_postprocess_batch(
        input_dir=detections_root,
        output_dir=pipeline_root,
        config=stage1_cfg,
        json_name="yolo_raw_detections.json",
        min_topology_detections=int(min_topology_detections),
        enable_fallback_detection=bool(enable_fallback_detection),
        save_individual_slices=False,
        save_debug=True,
    )

    n_total = len(samples)
    geo_pass_cnt = 0
    blank_pass_cnt = 0
    stage_success_cnt = 0
    reproj_mean_vals: List[float] = []
    reproj_max_vals: List[float] = []
    offset_max_vals: List[float] = []
    arm_margin_vals: List[float] = []
    blank_margin_vals: List[float] = []
    arm_top2_margin_vals: List[float] = []
    blank_top2_margin_vals: List[float] = []

    blank_acc_values: List[int] = []
    ref_acc_values: List[int] = []
    blank_gt_available = 0
    ref_gt_available = 0

    geometry_fail_cases: List[str] = []
    blank_unresolved_cases: List[str] = []
    pose_ambiguous_cases: List[str] = []

    for sample in samples:
        run_dir = pipeline_root / sample.chip_id
        topo = _read_json(run_dir / "debug_stage1_topology.json")
        geom = _read_json(run_dir / "debug_geometry_alignment.json")
        stage_meta = _read_json(run_dir / "stage1_metadata.json")

        final_status = str(topo.get("final_status", topo.get("status", "failed"))).lower()
        if final_status == "success":
            stage_success_cnt += 1

        qc = topo.get("qc", {}) if isinstance(topo.get("qc"), dict) else {}
        blank = topo.get("blank", {}) if isinstance(topo.get("blank"), dict) else {}

        # geo pass
        topo_fit_pass = bool(qc.get("fit_success", False))
        reproj_mean = _safe_float(
            geom.get("reprojection_error_mean_px", qc.get("reprojection_error_mean_px"))
        )
        reproj_max = _safe_float(
            geom.get("reprojection_error_max_px", qc.get("reprojection_error_max_px"))
        )
        reproj_thr = _safe_float(
            geom.get("reprojection_error_threshold_px", qc.get("reprojection_error_threshold_px"))
        )
        center_offset_max = _safe_float(geom.get("slice_center_offset_max_px"))
        center_offset_thr = _safe_float(geom.get("slice_center_offset_threshold_px"), default=8.0)

        reproj_mean_vals.append(reproj_mean)
        reproj_max_vals.append(reproj_max)
        offset_max_vals.append(center_offset_max)

        geo_pass = bool(
            topo_fit_pass
            and np.isfinite(reproj_mean)
            and np.isfinite(center_offset_max)
            and (not np.isfinite(reproj_thr) or reproj_mean <= reproj_thr)
            and (not np.isfinite(center_offset_thr) or center_offset_max <= center_offset_thr)
        )
        if geo_pass:
            geo_pass_cnt += 1
        else:
            geometry_fail_cases.append(sample.chip_id)

        # blank pass
        blank_status = str(blank.get("blank_status_pred", blank.get("blank_status", qc.get("blank_status_pred", qc.get("blank_status", "unknown")))))
        blank_conf = _safe_float(blank.get("blank_confidence", qc.get("blank_confidence")))
        blank_spread = _safe_float(blank.get("blank_score_spread", qc.get("blank_score_spread")))
        blank_spread_gate = _safe_float(qc.get("blank_spread_gate"), default=float("nan"))
        ref_arm_pred = blank.get("reference_arm_pred", blank.get("reference_arm", qc.get("reference_arm_pred", qc.get("reference_arm"))))
        arm_margin = _safe_float(blank.get("arm_margin", qc.get("arm_margin_pred")))
        blank_margin = _safe_float(blank.get("blank_margin", qc.get("blank_margin_pred")))
        arm_margin_thr = _safe_float(qc.get("blank_arm_margin_thr"))
        blank_margin_thr = _safe_float(qc.get("blank_margin_thr"))
        arm_top2_margin = _extract_top2_margin(blank.get("arm_top2", []))
        blank_top2_margin = _extract_top2_margin(blank.get("blank_top2", []))

        arm_margin_vals.append(arm_margin)
        blank_margin_vals.append(blank_margin)
        arm_top2_margin_vals.append(arm_top2_margin)
        blank_top2_margin_vals.append(blank_top2_margin)

        arm_margin_ok = True
        if np.isfinite(arm_margin) and np.isfinite(arm_margin_thr):
            arm_margin_ok = bool(arm_margin >= arm_margin_thr)
        blank_margin_ok = True
        if np.isfinite(blank_margin) and np.isfinite(blank_margin_thr):
            blank_margin_ok = bool(blank_margin >= blank_margin_thr)
        blank_spread_ok = True
        if np.isfinite(blank_spread_gate) and np.isfinite(blank_spread):
            blank_spread_ok = bool(blank_spread >= blank_spread_gate)

        blank_pass = bool(
            ref_arm_pred is not None
            and blank_status != "unresolved"
            and np.isfinite(blank_conf)
            and blank_conf >= 0.01
            and blank_spread_ok
            and arm_margin_ok
            and blank_margin_ok
        )
        if blank_pass:
            blank_pass_cnt += 1
        else:
            blank_unresolved_cases.append(sample.chip_id)

        if (
            np.isfinite(arm_top2_margin)
            and arm_top2_margin < 0.03
        ) or (
            np.isfinite(blank_top2_margin)
            and blank_top2_margin < 0.02
        ):
            pose_ambiguous_cases.append(sample.chip_id)

        # optional gt-based accuracy
        blank_id_pred = _safe_int(blank.get("blank_id_pred", blank.get("blank_id")), default=-1)
        if sample.gt_blank_template_idx is not None and blank_id_pred >= 0:
            blank_gt_available += 1
            blank_acc_values.append(int(blank_id_pred == int(sample.gt_blank_template_idx)))

        if sample.gt_reference_arm is not None and ref_arm_pred is not None:
            ref_gt_available += 1
            ref_acc_values.append(int(str(ref_arm_pred) == str(sample.gt_reference_arm)))

        # copy key artifacts
        if sample.chip_id in geometry_fail_cases[: max(0, key_cases_limit)]:
            dst_dir = key_cases_root / "pipeline_geometry_fail" / f"conf_{_conf_tag(conf)}" / sample.chip_id
            _copy_if_exists(run_dir / "debug_stage1_topology.png", dst_dir / "debug_stage1_topology.png")
            _copy_if_exists(run_dir / "debug_overlay.png", dst_dir / "debug_overlay.png")
            _copy_if_exists(sample.image_path, dst_dir / sample.image_path.name)
        if sample.chip_id in blank_unresolved_cases[: max(0, key_cases_limit)]:
            dst_dir = key_cases_root / "pipeline_blank_unresolved" / f"conf_{_conf_tag(conf)}" / sample.chip_id
            _copy_if_exists(run_dir / "debug_stage1_topology.png", dst_dir / "debug_stage1_topology.png")
            _copy_if_exists(run_dir / "debug_blank_features.json", dst_dir / "debug_blank_features.json")
            _copy_if_exists(sample.image_path, dst_dir / sample.image_path.name)
        if sample.chip_id in pose_ambiguous_cases[: max(0, key_cases_limit)]:
            dst_dir = key_cases_root / "pipeline_pose_ambiguity" / f"conf_{_conf_tag(conf)}" / sample.chip_id
            _copy_if_exists(run_dir / "debug_stage1_topology.png", dst_dir / "debug_stage1_topology.png")
            _copy_if_exists(sample.image_path, dst_dir / sample.image_path.name)

    row = {
        "conf": float(conf),
        "n_images": int(n_total),
        "stage1_success_rate": _safe_rate(stage_success_cnt, n_total),
        "geo_pass_rate": _safe_rate(geo_pass_cnt, n_total),
        "blank_pass_rate": _safe_rate(blank_pass_cnt, n_total),
        "reprojection_error_mean_px_mean": _mean(reproj_mean_vals),
        "reprojection_error_mean_px_median": _median(reproj_mean_vals),
        "reprojection_error_mean_px_p95": _quantile(reproj_mean_vals, 95),
        "reprojection_error_max_px_mean": _mean(reproj_max_vals),
        "reprojection_error_max_px_median": _median(reproj_max_vals),
        "reprojection_error_max_px_p95": _quantile(reproj_max_vals, 95),
        "slice_center_offset_max_px_mean": _mean(offset_max_vals),
        "slice_center_offset_max_px_median": _median(offset_max_vals),
        "slice_center_offset_max_px_p95": _quantile(offset_max_vals, 95),
        "arm_margin_mean": _mean(arm_margin_vals),
        "arm_margin_median": _median(arm_margin_vals),
        "arm_margin_p95": _quantile(arm_margin_vals, 95),
        "blank_margin_mean": _mean(blank_margin_vals),
        "blank_margin_median": _median(blank_margin_vals),
        "blank_margin_p95": _quantile(blank_margin_vals, 95),
        "arm_top2_margin_mean": _mean(arm_top2_margin_vals),
        "blank_top2_margin_mean": _mean(blank_top2_margin_vals),
        "geometry_fail_cases": int(len(set(geometry_fail_cases))),
        "blank_unresolved_cases": int(len(set(blank_unresolved_cases))),
        "pose_ambiguous_cases": int(len(set(pose_ambiguous_cases))),
        "blank_gt_coverage": _safe_rate(blank_gt_available, n_total),
        "reference_arm_gt_coverage": _safe_rate(ref_gt_available, n_total),
        "blank_accuracy": _mean(blank_acc_values),
        "reference_arm_accuracy": _mean(ref_acc_values),
    }
    return row


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _choose_recommended_conf(det_rows: Sequence[Dict[str, Any]], pipe_rows: Sequence[Dict[str, Any]]) -> float:
    pipe_by_conf = {float(r["conf"]): r for r in pipe_rows}
    scored: List[Tuple[float, float, float, float, float, float]] = []
    for det in det_rows:
        conf = float(det["conf"])
        pipe = pipe_by_conf.get(conf, {})
        geo = _safe_float(pipe.get("geo_pass_rate"))
        blank_pass = _safe_float(pipe.get("blank_pass_rate"))
        p12 = _safe_float(det.get("p_n_det_total_ge_12"))
        blank_rec = _safe_float(det.get("blank_recall_iou50_at_conf"))
        lt8 = _safe_float(det.get("p_n_det_total_lt_8"))
        score = (
            (0.55 * geo if np.isfinite(geo) else 0.0)
            + (0.25 * blank_pass if np.isfinite(blank_pass) else 0.0)
            + (0.10 * p12 if np.isfinite(p12) else 0.0)
            + (0.10 * blank_rec if np.isfinite(blank_rec) else 0.0)
            - (0.15 * lt8 if np.isfinite(lt8) else 0.0)
        )
        scored.append((score, conf, geo, blank_pass, p12, blank_rec))
    if not scored:
        return 0.05
    scored.sort(key=lambda x: (x[0], x[2], x[3], x[4], x[5]), reverse=True)
    return float(scored[0][1])


def _recommend_min_detections(det_rows: Sequence[Dict[str, Any]], conf: float) -> int:
    row = next((r for r in det_rows if abs(float(r["conf"]) - float(conf)) < 1e-9), None)
    if row is None:
        return 8
    p12 = _safe_float(row.get("p_n_det_total_ge_12"))
    p10 = _safe_float(row.get("p_n_det_total_ge_10"))
    if np.isfinite(p12) and p12 >= 0.95:
        return 12
    if np.isfinite(p10) and p10 >= 0.90:
        return 10
    return 8


def _make_report(
    *,
    out_path: Path,
    weights_path: Path,
    data_yaml: Path,
    conf_list: Sequence[float],
    nms_iou: float,
    min_topology_detections: int,
    val_metrics: Dict[str, Any],
    det_rows: Sequence[Dict[str, Any]],
    pipe_rows: Sequence[Dict[str, Any]],
    recommended_conf: float,
    recommended_min_det: int,
) -> None:
    per_class = val_metrics.get("per_class", {})
    cls_lines: List[str] = []
    for cls_name, rec in per_class.items():
        cls_lines.append(
            f"- `{cls_name}`: AP50={_safe_float(rec.get('ap50')):.4f}, "
            f"AP50-95={_safe_float(rec.get('ap50_95')):.4f}, Recall={_safe_float(rec.get('recall')):.4f}"
        )
    if not cls_lines:
        cls_lines = ["- per-class metrics unavailable"]

    det_by_conf = {float(r["conf"]): r for r in det_rows}
    pipe_by_conf = {float(r["conf"]): r for r in pipe_rows}
    rec_det = det_by_conf.get(float(recommended_conf), {})
    rec_pipe = pipe_by_conf.get(float(recommended_conf), {})

    blank_recall = _safe_float(rec_det.get("blank_recall_iou50_at_conf"))
    blank_miss_lit_full = _safe_float(rec_det.get("blank_miss_lit_full_cases"), default=0.0)
    geo_pass = _safe_float(rec_pipe.get("geo_pass_rate"))
    blank_pass = _safe_float(rec_pipe.get("blank_pass_rate"))
    ref_acc = _safe_float(rec_pipe.get("reference_arm_accuracy"))
    blank_acc = _safe_float(rec_pipe.get("blank_accuracy"))

    prefer_single_cls = False
    if np.isfinite(blank_recall) and blank_recall < 0.90:
        prefer_single_cls = True
    if np.isfinite(blank_miss_lit_full) and blank_miss_lit_full > 0:
        prefer_single_cls = True
    if np.isfinite(blank_pass) and blank_pass < 0.90:
        prefer_single_cls = True

    compare_dual_single = (
        "当前只完成了双类别模型实测；单类别模型训练完成后可用同一脚本复跑，自动填充同口径对照。"
    )
    if prefer_single_cls:
        conclusion = (
            "从当前双类别结果看，Stage1 更建议偏向 single-class 检测：BLANK 由后处理语义门控识别，"
            "可避免 BLANK/LIT 分类错分导致的掉检传播。"
        )
    else:
        conclusion = (
            "从当前双类别结果看，双类方案在 BLANK 召回与 pipeline 通过率上可接受，"
            "暂不需要强制切到 single-class；建议待 single-class 训练完成后做同口径复核。"
        )

    lines = [
        "# Stage1 Ablation Report",
        "",
        "## Run Setup",
        f"- weights: `{weights_path}`",
        f"- data: `{data_yaml}`",
        f"- conf grid: `{', '.join(f'{c:.3f}' for c in conf_list)}`",
        f"- NMS IoU: `{nms_iou}`",
        f"- min_topology_detections (input): `{min_topology_detections}`",
        "",
        "## Task A: Detection-only (per-class)",
        *cls_lines,
        "",
        "## Task B: Pipeline (topology + canonical slicing)",
        f"- recommended conf row geo_pass: `{geo_pass:.4f}`",
        f"- recommended conf row blank_pass: `{blank_pass:.4f}`",
        f"- reference_arm accuracy (if inferable from GT geometry): `{ref_acc:.4f}`",
        f"- blank accuracy (if inferable from GT geometry): `{blank_acc:.4f}`",
        "",
        "## Task C: Single-class vs Dual-class",
        f"- {conclusion}",
        f"- {compare_dual_single}",
        "",
        "## Recommended Stage1 Config",
        f"- `conf`: `{recommended_conf:.3f}`",
        f"- `nms_iou`: `{nms_iou:.2f}`",
        f"- `min_topology_detections`: `{recommended_min_det}`",
        "",
        "## Notes",
        "- `key_cases/` 已按 BLANK掉检、误检、几何失败、姿态歧义分类导出 overlay。",
        "- `detection_ablation_table.csv` 与 `pipeline_ablation_table.csv` 可直接用于论文表格整理。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 ablation runner (detection + pipeline)")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--template", type=Path, default=Path("configs/templates/pinwheel_v3_centered.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/yolo/chambers_v1/ablation_stage1"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--conf-list", type=str, default="0.01,0.03,0.05,0.1,0.2")
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--min-topology-detections", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--enable-fallback-detection", action="store_true")
    parser.add_argument("--key-cases-limit", type=int, default=30)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conf_list = _parse_conf_list(args.conf_list)
    image_paths, labels_dir, class_names = _resolve_dataset_paths(args.data_yaml.resolve(), split=args.split)
    samples = _load_gt_samples(image_paths=image_paths, labels_dir=labels_dir)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: int(args.max_samples)]

    cfg = load_config_from_yaml(args.config.resolve())
    stage1_cfg = cfg.stage1.model_copy(deep=True)
    stage1_cfg.yolo.weights_path = str(args.weights.resolve())
    stage1_cfg.yolo.device = str(args.device)
    if stage1_cfg.topology is None:
        stage1_cfg.topology = TopologyConfig()
    if args.template is not None:
        stage1_cfg.topology.template_path = str(args.template.resolve())

    print(f"[INFO] samples={len(samples)} split={args.split}")
    print(f"[INFO] weights={args.weights.resolve()}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] class_names={class_names}")

    model = YOLO(str(args.weights.resolve()))
    val_metrics = _run_val_per_class_metrics(
        model=model,
        data_yaml=args.data_yaml.resolve(),
        split=args.split,
        imgsz=args.imgsz,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        device=args.device,
    )
    (output_dir / "val_per_class_metrics.json").write_text(
        json.dumps(val_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[INFO] val per-class metrics done")

    all_preds = _predict_all_images(
        model=model,
        samples=samples,
        min_conf=min(conf_list) * 0.5,
        imgsz=args.imgsz,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        device=args.device,
    )
    print("[INFO] low-conf predictions done")

    # optional GT mapping for blank/reference_arm accuracy
    _prepare_gt_template_mapping(
        samples=samples,
        stage1_topology=stage1_cfg.topology,
        template_path=args.template.resolve() if args.template is not None else None,
    )

    detection_rows: List[Dict[str, Any]] = []
    pipeline_rows: List[Dict[str, Any]] = []
    key_cases_root = output_dir / "key_cases"

    for conf in conf_list:
        print(f"[INFO] conf={conf:.3f} start")
        conf_tag = _conf_tag(conf)
        conf_root = output_dir / f"conf_{conf_tag}"
        det_json_root = conf_root / "detections"
        pipe_root = conf_root / "pipeline_runs"

        det_row, _, _ = _evaluate_detection_conf(
            conf=conf,
            samples=samples,
            all_preds=all_preds,
            val_metrics=val_metrics,
            det_json_root=det_json_root,
            key_cases_root=key_cases_root,
            key_cases_limit=args.key_cases_limit,
        )
        detection_rows.append(det_row)

        pipe_row = _evaluate_pipeline_conf(
            conf=conf,
            samples=samples,
            detections_root=det_json_root,
            pipeline_root=pipe_root,
            stage1_cfg=stage1_cfg,
            min_topology_detections=int(args.min_topology_detections),
            enable_fallback_detection=bool(args.enable_fallback_detection),
            key_cases_root=key_cases_root,
            key_cases_limit=args.key_cases_limit,
        )
        pipeline_rows.append(pipe_row)
        print(f"[INFO] conf={conf:.3f} done")

    detection_csv = output_dir / "detection_ablation_table.csv"
    pipeline_csv = output_dir / "pipeline_ablation_table.csv"
    _write_csv(detection_csv, detection_rows)
    _write_csv(pipeline_csv, pipeline_rows)

    recommended_conf = _choose_recommended_conf(detection_rows, pipeline_rows)
    recommended_min_det = _recommend_min_detections(detection_rows, conf=recommended_conf)
    report_md = output_dir / "ablation_report.md"
    _make_report(
        out_path=report_md,
        weights_path=args.weights.resolve(),
        data_yaml=args.data_yaml.resolve(),
        conf_list=conf_list,
        nms_iou=float(args.nms_iou),
        min_topology_detections=int(args.min_topology_detections),
        val_metrics=val_metrics,
        det_rows=detection_rows,
        pipe_rows=pipeline_rows,
        recommended_conf=recommended_conf,
        recommended_min_det=recommended_min_det,
    )

    print("[INFO] outputs:")
    print(f"  - {detection_csv}")
    print(f"  - {pipeline_csv}")
    print(f"  - {report_md}")
    print(f"  - {key_cases_root}")
    print(f"[INFO] recommended conf={recommended_conf:.3f}, min_topology_detections={recommended_min_det}")


if __name__ == "__main__":
    main()
