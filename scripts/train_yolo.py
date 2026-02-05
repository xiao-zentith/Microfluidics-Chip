#!/usr/bin/env python3
"""
YOLO 目标检测训练脚本

自动处理路径问题，避免受 YOLO 全局配置 (~/.config/Ultralytics/settings.json) 影响。

用法:
    # 查看帮助
    python scripts/train_yolo.py --help
    
    # 基本训练
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v1
    
    # 自定义参数
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v2 --epochs 100 --imgsz 1280 --batch 16
    
    # 继续训练
    python scripts/train_yolo.py --resume data/experiments/yolo/chambers_v1/weights/last.pt
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_absolute_path(relative_path: str) -> str:
    """将相对路径转换为绝对路径（相对于项目根目录）"""
    return str(PROJECT_ROOT / relative_path)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 目标检测训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本训练
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v1
    
    # 自定义参数 (高分辨率)
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v2 \\
        --epochs 100 --imgsz 1280 --batch 16 --model yolo11n.pt
    
    # 继续训练
    python scripts/train_yolo.py --resume data/experiments/yolo/chambers_v1/weights/last.pt
"""
    )
    
    # 数据集配置
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="yolo_v3",
        help="数据集名称 (在 data/stage1_detection/ 下的目录名), 默认: yolo_v3"
    )
    
    # 模型配置
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11n.pt",
        help="预训练模型, 默认: yolo11n.pt"
    )
    
    # 训练参数
    parser.add_argument("--epochs", "-e", type=int, default=100, help="训练轮数, 默认: 100")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸, 默认: 640")
    parser.add_argument("--batch", "-b", type=int, default=16, help="批次大小, 默认: 16")
    parser.add_argument("--device", type=str, default="0", help="设备 (0, 1, cpu), 默认: 0")
    parser.add_argument("--name", "-n", type=str, default="train", help="实验名称, 默认: train")
    
    # 数据增强参数
    parser.add_argument("--hsv_h", type=float, default=0.015, help="色调增强, 默认: 0.015")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="饱和度增强, 默认: 0.7")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="亮度增强, 默认: 0.4")
    parser.add_argument("--degrees", type=float, default=10.0, help="旋转角度, 默认: 10.0")
    parser.add_argument("--translate", type=float, default=0.1, help="平移, 默认: 0.1")
    parser.add_argument("--scale", type=float, default=0.5, help="缩放, 默认: 0.5")
    parser.add_argument("--flipud", type=float, default=0.0, help="垂直翻转概率, 默认: 0.0")
    parser.add_argument("--fliplr", type=float, default=0.5, help="水平翻转概率, 默认: 0.5")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic增强, 默认: 1.0")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp增强, 默认: 0.0")
    
    # 继续训练
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点继续训练 (提供 last.pt 的路径)"
    )
    
    args = parser.parse_args()
    
    # 延迟导入 ultralytics (可能较慢)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)
    
    # 构建数据集路径 (使用绝对路径)
    data_yaml = get_absolute_path(f"data/stage1_detection/{args.data}/data.yaml")
    if not Path(data_yaml).exists():
        print(f"错误: 数据集配置不存在: {data_yaml}")
        print(f"可用数据集: {list(Path(get_absolute_path('data/stage1_detection')).iterdir())}")
        sys.exit(1)
    
    # 构建输出路径 (使用绝对路径，避免全局配置干扰)
    project_dir = get_absolute_path("data/experiments/yolo")
    
    print("=" * 60)
    print("YOLO 训练配置")
    print("=" * 60)
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  数据集配置: {data_yaml}")
    print(f"  输出目录:   {project_dir}/{args.name}")
    print(f"  模型:       {args.model}")
    print(f"  轮数:       {args.epochs}")
    print(f"  图像尺寸:   {args.imgsz}")
    print(f"  批次大小:   {args.batch}")
    print(f"  设备:       {args.device}")
    print("=" * 60)
    
    # 继续训练模式
    if args.resume:
        resume_path = get_absolute_path(args.resume) if not Path(args.resume).is_absolute() else args.resume
        if not Path(resume_path).exists():
            print(f"错误: 检查点不存在: {resume_path}")
            sys.exit(1)
        print(f"从检查点继续训练: {resume_path}")
        model = YOLO(resume_path)
        results = model.train(resume=True)
    else:
        # 新建训练
        model = YOLO(args.model)
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=project_dir,  # 使用绝对路径
            name=args.name,
            # 数据增强
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,
        )
    
    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
    if hasattr(results, 'box') and results.box is not None:
        print(f"  mAP@0.5:    {results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  结果保存在: {project_dir}/{args.name}")
    print("=" * 60)
    
    # 提示后续操作
    print("\n后续操作:")
    print(f"  1. 查看训练曲线: {project_dir}/{args.name}/results.png")
    print(f"  2. 复制最佳模型: cp {project_dir}/{args.name}/weights/best.pt weights/yolo/best.pt")


if __name__ == "__main__":
    main()
