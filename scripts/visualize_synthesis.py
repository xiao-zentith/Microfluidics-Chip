"""
数据增强可视化脚本

展示合成数据生成过程的各个阶段，帮助理解和调试数据增强策略。

使用方法：
  python scripts/visualize_synthesis.py \
    dataset/clean_images/chip001.png \
    -o viz/synthesis_demo.png
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microfluidics_chip.core.config import get_default_config, load_config_from_yaml
from microfluidics_chip.stage1_detection.detector import ChamberDetector
from microfluidics_chip.stage1_detection.synthesizer import FullChipSynthesizer
from microfluidics_chip.core.logger import setup_logger, get_logger

logger = get_logger("visualize_synthesis")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize data synthesis/augmentation process",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "input_image",
        type=Path,
        help="输入清洁GT图像路径"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("viz/synthesis_visualization.png"),
        help="输出可视化图像路径（默认: viz/synthesis_visualization.png）"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="配置文件路径（可选）"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细日志"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger(level="DEBUG" if args.verbose else "INFO")
    
    if not args.input_image.exists():
        logger.error(f"Input image not found: {args.input_image}")
        return 1
    
    # 加载配置
    if args.config and args.config.exists():
        config = load_config_from_yaml(args.config)
    else:
        config = get_default_config()
    
    # 初始化检测器和合成器
    logger.info("Initializing YOLO detector...")
    detector = ChamberDetector(config.stage1.yolo)
    
    logger.info("Initializing synthesizer...")
    synthesizer = FullChipSynthesizer(
        detector=detector,
        geometry_config=config.stage1.geometry,
        class_id_blank=config.stage1.geometry.class_id_blank
    )
    
    # 生成可视化
    logger.info(f"Processing image: {args.input_image}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    synthesizer.visualize_synthesis_process(
        clean_full_img_path=args.input_image,
        save_path=args.output
    )
    
    logger.info(f"✓ Visualization saved to: {args.output}")
    logger.info("\n可视化面板说明:")
    logger.info("  1. Original Clean     - 原始清洁图像")
    logger.info("  2. Concentration Mask - 浓度掩膜（仅腔室区域）")
    logger.info("  3. Virtual Concentration - 颜色抖动效果")
    logger.info("  4. Concentration Change - 浓度变化差异")
    logger.info("  5. Final Dirty Output - 最终合成结果（含所有降质）")
    logger.info("  6. Total Degradation  - 总体降质程度")
    
    return 0


if __name__ == "__main__":
    exit(main())
