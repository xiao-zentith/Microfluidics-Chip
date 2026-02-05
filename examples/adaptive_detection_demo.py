#!/usr/bin/env python3
"""
自适应检测端到端示例

演示从原图输入到输出 12 个腔室坐标（含暗腔室回填）的完整流程。

用法:
    # 基本用法
    python examples/adaptive_detection_demo.py --image path/to/image.jpg
    
    # 指定权重和输出目录
    python examples/adaptive_detection_demo.py \
        --image path/to/image.jpg \
        --weights weights/yolo/best.pt \
        --output output/demo/
    
    # 使用配置文件
    python examples/adaptive_detection_demo.py \
        --image path/to/image.jpg \
        --config configs/adaptive_detection.yaml
"""

import argparse
import json
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加 src 到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microfluidics_chip.core.config import (
    YOLOConfig, 
    GeometryConfig, 
    Stage1Config,
    AdaptiveDetectionConfig,
    TopologyConfig
)
from microfluidics_chip.core.logger import setup_logger, get_logger
from microfluidics_chip.stage1_detection.inference import infer_stage1_adaptive

logger = get_logger("adaptive_demo")


def draw_detection_result(
    image: np.ndarray,
    result,
    output_path: Path
) -> np.ndarray:
    """
    绘制检测结果可视化
    
    :param image: 原始图像
    :param result: AdaptiveDetectionResult
    :param output_path: 输出路径
    :return: 可视化图像
    """
    vis = image.copy()
    
    # 绘制 ROI 边界
    x1, y1, x2, y2 = result.roi_bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(vis, f"ROI (score={result.cluster_score:.2f})", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 绘制所有拟合的腔室位置
    for i, (cx, cy) in enumerate(result.fitted_centers):
        cx, cy = int(cx), int(cy)
        
        # 根据状态选择颜色
        if not result.visibility[i]:
            # 不可见：灰色虚线圆
            color = (128, 128, 128)
            thickness = 1
        elif i in result.dark_chamber_indices:
            # 暗腔室：红色
            color = (0, 0, 255)
            thickness = 3
        elif result.detected_mask[i]:
            # 检测到的：绿色
            color = (0, 255, 0)
            thickness = 2
        else:
            # 回填的：黄色
            color = (0, 255, 255)
            thickness = 2
        
        cv2.circle(vis, (cx, cy), 20, color, thickness)
        cv2.putText(vis, str(i), (cx - 5, cy + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 添加图例
    legend_y = 30
    cv2.putText(vis, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 25
    cv2.circle(vis, (20, legend_y), 8, (0, 255, 0), -1)
    cv2.putText(vis, "Detected", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 25
    cv2.circle(vis, (20, legend_y), 8, (0, 255, 255), -1)
    cv2.putText(vis, "Filled (missing)", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 25
    cv2.circle(vis, (20, legend_y), 8, (0, 0, 255), -1)
    cv2.putText(vis, "Dark chamber", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 25
    cv2.circle(vis, (20, legend_y), 8, (128, 128, 128), -1)
    cv2.putText(vis, "Out of view", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 保存
    cv2.imwrite(str(output_path), vis)
    logger.info(f"Visualization saved: {output_path}")
    
    return vis


def load_config_from_yaml(yaml_path: Path):
    """从 YAML 加载自适应检测配置"""
    import yaml
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    adaptive_config = None
    topology_config = None
    
    if 'adaptive_detection' in data:
        adaptive_config = AdaptiveDetectionConfig(**data['adaptive_detection'])
    
    if 'topology' in data:
        topology_config = TopologyConfig(**data['topology'])
    
    return adaptive_config, topology_config


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Detection End-to-End Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--image", "-i", type=Path, required=True,
                        help="输入图像路径")
    parser.add_argument("--weights", "-w", type=Path, 
                        default=PROJECT_ROOT / "weights" / "yolo" / "best.pt",
                        help="YOLO 权重路径")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="输出目录 (默认: output/adaptive_demo/)")
    parser.add_argument("--config", "-c", type=Path, default=None,
                        help="自适应检测配置文件 (YAML)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    setup_logger(level="INFO")
    
    # 验证输入
    if not args.image.exists():
        logger.error(f"Image not found: {args.image}")
        return 1
    
    if not args.weights.exists():
        logger.error(f"Weights not found: {args.weights}")
        return 1
    
    # 设置输出目录
    if args.output is None:
        args.output = PROJECT_ROOT / "output" / "adaptive_demo"
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Adaptive Detection Demo")
    logger.info("=" * 60)
    logger.info(f"Input:   {args.image}")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Output:  {args.output}")
    
    # ==================== 加载配置 ====================
    adaptive_config = AdaptiveDetectionConfig()
    topology_config = TopologyConfig()
    
    if args.config and args.config.exists():
        logger.info(f"Loading config from: {args.config}")
        adaptive_config, topology_config = load_config_from_yaml(args.config)
        if adaptive_config is None:
            adaptive_config = AdaptiveDetectionConfig()
        if topology_config is None:
            topology_config = TopologyConfig()
    
    # ==================== 加载图像 ====================
    image = cv2.imread(str(args.image))
    if image is None:
        logger.error(f"Failed to read image: {args.image}")
        return 1
    
    logger.info(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # ==================== 创建配置 ====================
    stage1_config = Stage1Config(
        yolo=YOLOConfig(
            weights_path=str(args.weights),
            device=args.device,
            confidence_threshold=0.3
        ),
        geometry=GeometryConfig()
    )
    
    # ==================== 运行推理 ====================
    logger.info("Starting adaptive inference...")
    
    chip_id = args.image.stem
    
    try:
        result = infer_stage1_adaptive(
            chip_id=chip_id,
            raw_image=image,
            config=stage1_config,
            adaptive_config=adaptive_config,
            topology_config=topology_config
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ==================== 输出结果 ====================
    logger.info("=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    logger.info(f"Detected chambers:    {len(result.detections)}")
    logger.info(f"Fit success:          {result.fit_success}")
    logger.info(f"Inlier ratio:         {result.inlier_ratio:.2f}")
    logger.info(f"Reprojection error:   {result.reprojection_error:.2f} px")
    logger.info(f"Dark chamber indices: {result.dark_chamber_indices}")
    logger.info(f"Processing time:      {result.processing_time:.2f}s")
    
    # 输出所有 12 个腔室坐标
    logger.info("\nAll 12 chamber coordinates:")
    for i, (cx, cy) in enumerate(result.fitted_centers):
        status = []
        if result.visibility[i]:
            if result.detected_mask[i]:
                status.append("detected")
            else:
                status.append("filled")
            if i in result.dark_chamber_indices:
                status.append("dark")
        else:
            status.append("out-of-view")
        
        logger.info(f"  Chamber {i:2d}: ({cx:7.1f}, {cy:7.1f}) [{', '.join(status)}]")
    
    # ==================== 保存结果 ====================
    # 1. 可视化
    vis_path = args.output / f"{chip_id}_visualization.png"
    draw_detection_result(image, result, vis_path)
    
    # 2. JSON 结果
    json_result = {
        "chip_id": chip_id,
        "timestamp": datetime.now().isoformat(),
        "detected_count": len(result.detections),
        "fit_success": result.fit_success,
        "inlier_ratio": result.inlier_ratio,
        "reprojection_error": result.reprojection_error,
        "dark_chamber_indices": result.dark_chamber_indices,
        "processing_time": result.processing_time,
        "chambers": [
            {
                "index": i,
                "center": [float(cx), float(cy)],
                "visible": bool(result.visibility[i]),
                "detected": bool(result.detected_mask[i]),
                "is_dark": i in result.dark_chamber_indices
            }
            for i, (cx, cy) in enumerate(result.fitted_centers)
        ]
    }
    
    json_path = args.output / f"{chip_id}_detections.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {json_path}")
    
    logger.info("=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
