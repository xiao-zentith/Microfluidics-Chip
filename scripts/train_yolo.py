#!/usr/bin/env python3
"""
YOLO 目标检测训练脚本

自动处理路径问题，避免受 YOLO 全局配置 (~/.config/Ultralytics/settings.json) 影响。

用法:
    # 查看帮助
    python scripts/train_yolo.py --help
    
    # 基本训练
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v1

    # 单卡训练（指定 GPU0）
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v1 --mode single --gpus 0

    # 多卡训练（自动使用全部可见 GPU）
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v1 --mode multi --gpus all
    
    # 无验证集训练（val 缺失时推荐）
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v1 --no-val

    # 自定义参数
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v2 --epochs 100 --imgsz 1280 --batch 16
    
    # 继续训练
    python scripts/train_yolo.py --resume data/experiments/yolo/chambers_v1/weights/last.pt
"""

import argparse
import sys
from pathlib import Path
import yaml
from typing import Optional, Tuple, List

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_absolute_path(relative_path: str) -> str:
    """将相对路径转换为绝对路径（相对于项目根目录）"""
    return str(PROJECT_ROOT / relative_path)


def resolve_dataset_split_path(data_yaml: Path, split_path: str) -> Path:
    """解析 data.yaml 中的 split 路径为绝对路径。"""
    p = Path(split_path)
    if p.is_absolute():
        return p
    return data_yaml.parent / p


def count_images_in_dir(path: Path) -> int:
    """统计目录中的图像文件数量（非递归）。"""
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_device(device_arg: str) -> Tuple[str, Optional[int]]:
    """
    解析训练设备参数。

    - auto: 自动使用所有可见 GPU；无 GPU 时回退 CPU
    - 其它输入: 原样返回（如 "0"、"0,1"、"cpu"）
    """
    normalized = str(device_arg).strip().lower()
    if normalized != "auto":
        if normalized == "cpu":
            return "cpu", 0
        if "," in normalized:
            count = len([x for x in normalized.split(",") if x.strip() != ""])
            return device_arg, count
        if normalized.isdigit():
            return device_arg, 1
        return device_arg, None

    try:
        import torch
    except Exception:
        return "cpu", 0

    if not torch.cuda.is_available():
        return "cpu", 0

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return "0", gpu_count

    all_gpus = ",".join(str(i) for i in range(gpu_count))
    return all_gpus, gpu_count


def get_visible_gpu_ids() -> List[int]:
    """返回当前进程可见的 GPU id 列表（基于 torch 可见设备）。"""
    try:
        import torch
    except Exception:
        return []

    if not torch.cuda.is_available():
        return []

    return list(range(torch.cuda.device_count()))


def parse_gpu_selector(gpus_arg: str, visible_gpu_ids: List[int]) -> List[int]:
    """
    解析 --gpus 参数。

    支持:
    - all
    - 逗号分隔 id，如 0,1
    """
    if not visible_gpu_ids:
        return []

    if str(gpus_arg).strip().lower() == "all":
        return visible_gpu_ids.copy()

    selected: List[int] = []
    for token in str(gpus_arg).split(","):
        token = token.strip()
        if token == "":
            continue
        if not token.isdigit():
            raise ValueError(f"非法 GPU id: {token}")
        gpu_id = int(token)
        if gpu_id not in visible_gpu_ids:
            raise ValueError(f"GPU id {gpu_id} 不在可见范围 {visible_gpu_ids}")
        if gpu_id not in selected:
            selected.append(gpu_id)

    return selected


def resolve_training_device(
    mode: str,
    gpus_arg: str,
    device_override: Optional[str]
) -> Tuple[str, int, str]:
    """
    解析最终训练 device 与使用卡数。

    优先级:
    1) --device 覆盖（兼容 Ultralytics 原生参数）
    2) --mode + --gpus
    """
    if device_override is not None:
        resolved, count = resolve_device(device_override)
        return resolved, count or 0, f"override({device_override})"

    mode = mode.lower().strip()
    visible_gpu_ids = get_visible_gpu_ids()

    if mode == "cpu":
        return "cpu", 0, "mode=cpu"

    if not visible_gpu_ids:
        if mode in {"single", "multi"}:
            raise ValueError("当前无可见 GPU，无法使用 single/multi 模式。")
        return "cpu", 0, "auto-fallback-cpu"

    selected = parse_gpu_selector(gpus_arg, visible_gpu_ids)
    if not selected:
        selected = visible_gpu_ids.copy()

    if mode == "single":
        selected = [selected[0]]
    elif mode == "multi":
        if len(selected) < 2:
            raise ValueError(
                f"multi 模式需要至少 2 张 GPU，当前选择为 {selected}。"
            )
    elif mode != "auto":
        raise ValueError(f"不支持的 mode: {mode}")
    # auto: 保持 selected 原样（默认 all）

    device = ",".join(str(x) for x in selected) if len(selected) > 1 else str(selected[0])
    return device, len(selected), f"mode={mode}, gpus={selected}"


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 目标检测训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本训练
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v1

    # 单卡训练（GPU0）
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v1 --mode single --gpus 0

    # 多卡训练（全部 GPU），并按每卡 batch 自动计算总 batch
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v1 \\
        --mode multi --gpus all --batch-per-gpu 8
    
    # 无验证集训练
    python scripts/train_yolo.py --data yolo_v3 --name chambers_v1 --no-val

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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "single", "multi", "cpu"],
        default="auto",
        help="训练模式: auto/single/multi/cpu, 默认: auto"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="all",
        help="GPU 选择: all 或逗号分隔 id (如 0,1), 默认: all"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="高级覆盖参数（直接传给 Ultralytics device，如 0,1 或 cpu），优先级最高"
    )
    parser.add_argument(
        "--batch-per-gpu",
        type=int,
        default=None,
        help="按每卡 batch 自动计算总 batch（总batch = 每卡batch x GPU数量）"
    )
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
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="无验证集模式：跳过验证。若 val 缺失/为空也会自动启用。"
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

    # 训练前检查数据 split，并在必要时自动切换到无验证集模式
    data_yaml_path = Path(data_yaml)
    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"错误: 无法读取数据集配置: {data_yaml_path}")
        print(f"详情: {e}")
        sys.exit(1)

    train_cfg = data_cfg.get("train")
    if not train_cfg:
        print(f"错误: data.yaml 缺少 train 配置: {data_yaml_path}")
        sys.exit(1)

    train_path = resolve_dataset_split_path(data_yaml_path, str(train_cfg))
    if not train_path.exists():
        print(f"错误: data.yaml 中的 train 路径不存在: {train_path}")
        sys.exit(1)

    train_count = count_images_in_dir(train_path)
    if train_count <= 0:
        print(f"错误: train 目录没有可用图像: {train_path}")
        sys.exit(1)

    no_val_mode = args.no_val
    val_cfg = data_cfg.get("val")
    val_count = 0
    val_path = None
    if val_cfg:
        val_path = resolve_dataset_split_path(data_yaml_path, str(val_cfg))
        val_count = count_images_in_dir(val_path)

    if not no_val_mode:
        if not val_cfg or val_path is None or not val_path.exists() or val_count <= 0:
            no_val_mode = True
            print("警告: 未检测到可用验证集，自动切换到无验证集训练模式 (--no-val)。")
            print("      当前训练将跳过每轮验证，建议后续补标后再做正式评估。")

    effective_data_yaml = data_yaml_path
    if no_val_mode:
        # Ultralytics 的数据检查需要 val 字段存在；无验证时临时指向 train 并关闭 val 流程
        data_cfg["val"] = str(train_cfg)
        effective_data_yaml = data_yaml_path.parent / f"{data_yaml_path.stem}.train_only.yaml"
        with open(effective_data_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)

    try:
        resolved_device, gpu_count, device_strategy = resolve_training_device(
            mode=args.mode,
            gpus_arg=args.gpus,
            device_override=args.device
        )
    except ValueError as e:
        print(f"错误: 设备参数无效: {e}")
        sys.exit(1)

    effective_batch = args.batch
    if args.batch_per_gpu is not None:
        if args.batch_per_gpu <= 0:
            print("错误: --batch-per-gpu 必须为正整数")
            sys.exit(1)
        if gpu_count > 0:
            effective_batch = args.batch_per_gpu * gpu_count
        else:
            effective_batch = args.batch_per_gpu
    
    # 构建输出路径 (使用绝对路径，避免全局配置干扰)
    project_dir = get_absolute_path("data/experiments/yolo")
    
    print("=" * 60)
    print("YOLO 训练配置")
    print("=" * 60)
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  数据集配置: {effective_data_yaml}")
    print(f"  输出目录:   {project_dir}/{args.name}")
    print(f"  模型:       {args.model}")
    print(f"  轮数:       {args.epochs}")
    print(f"  图像尺寸:   {args.imgsz}")
    print(f"  批次大小:   {effective_batch}")
    print(f"  设备:       {resolved_device}")
    print(f"  设备策略:   {device_strategy}")
    print(f"  训练集样本: {train_count}")
    if no_val_mode:
        print("  验证模式:   关闭 (no-val)")
    else:
        print(f"  验证集样本: {val_count}")
    if gpu_count > 1:
        print(f"  GPU数量:    {gpu_count} (多卡并行)")
    print("=" * 60)
    
    # 继续训练模式
    if args.resume:
        resume_path = get_absolute_path(args.resume) if not Path(args.resume).is_absolute() else args.resume
        if not Path(resume_path).exists():
            print(f"错误: 检查点不存在: {resume_path}")
            sys.exit(1)
        print(f"从检查点继续训练: {resume_path}")
        model = YOLO(resume_path)
        results = model.train(resume=True, device=resolved_device)
    else:
        # 新建训练
        model = YOLO(args.model)
        results = model.train(
            data=str(effective_data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=effective_batch,
            device=resolved_device,
            project=project_dir,  # 使用绝对路径
            name=args.name,
            val=not no_val_mode,
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
