"""
端到端示例：Stage1 + Stage2 完整流程

演示如何使用 Python API 处理微流控芯片图像
"""

from pathlib import Path
from microfluidics_chip.core.config import load_config_from_yaml, get_default_config
from microfluidics_chip.pipelines.stage1 import run_stage1, run_stage1_batch
from microfluidics_chip.pipelines.stage2 import run_stage2, run_stage2_batch


def example_single_chip():
    """示例 1: 处理单个芯片"""
    print("=" * 60)
    print("示例 1: 处理单个芯片")
    print("=" * 60)
    
    # 1. 加载配置
    config = get_default_config()
    # 或从文件加载: config = load_config_from_yaml(Path("configs/default.yaml"))
    
    # 2. Stage1: 检测与切片
    print("\n[Stage1] 检测与切片...")
    stage1_output = run_stage1(
        chip_id="chip001",
        raw_image_path=Path("data/chip001.png"),
        gt_image_path=None,  # 可选
        output_dir=Path("runs/demo/stage1"),
        config=config.stage1,
        save_individual_slices=True,  # 保存单个切片用于调试
        save_debug=True  # 保存检测可视化
    )
    
    print(f"Stage1 完成！")
    print(f"  - 检测到腔室数: {stage1_output.num_chambers}")
    print(f"  - 处理时间: {stage1_output.processing_time:.2f}s")
    print(f"  - 输出目录: runs/demo/stage1/chip001/")
    
    # 3. Stage2: 光照校正
    print("\n[Stage2] 光照校正...")
    stage2_output = run_stage2(
        stage1_run_dir=Path("runs/demo/stage1/chip001"),  # P2: 只接受 stage1_run_dir
        output_dir=Path("runs/demo/stage2"),
        config=config.stage2
    )
    
    print(f"Stage2 完成！")
    print(f"  - 处理时间: {stage2_output.processing_time:.2f}s")
    print(f"  - 输出目录: runs/demo/stage2/chip001/")
    
    print("\n✓ 单芯片处理完成！")


def example_batch_processing():
    """示例 2: 批量处理"""
    print("=" * 60)
    print("示例 2: 批量处理")
    print("=" * 60)
    
    # 加载配置
    config = get_default_config()
    
    # Stage1 批量
    print("\n[Stage1 Batch] 批量检测与切片...")
    stage1_outputs = run_stage1_batch(
        input_dir=Path("data/raw_images"),
        output_dir=Path("runs/batch/stage1"),
        config=config.stage1,
        gt_suffix="_gt"
    )
    
    print(f"Stage1 批量完成: {len(stage1_outputs)} 个芯片")
    
    # Stage2 批量
    print("\n[Stage2 Batch] 批量光照校正...")
    stage2_outputs = run_stage2_batch(
        stage1_output_dir=Path("runs/batch/stage1"),
        output_dir=Path("runs/batch/stage2"),
        config=config.stage2
    )
    
    print(f"Stage2 批量完成: {len(stage2_outputs)} 个芯片")
    print("\n✓ 批量处理完成！")


def example_custom_config():
    """示例 3: 使用自定义配置"""
    print("=" * 60)
    print("示例 3: 使用自定义配置")
    print("=" * 60)
    
    # 从自定义配置文件加载
    config = load_config_from_yaml(Path("configs/my_config.yaml"))
    
    # 处理芯片
    stage1_output = run_stage1(
        chip_id="chip_custom",
        raw_image_path=Path("data/chip_custom.png"),
        gt_image_path=None,
        output_dir=Path("runs/custom/stage1"),
        config=config.stage1
    )
    
    print(f"✓ 使用自定义配置处理完成！")


def example_with_gt():
    """示例 4: 带 Ground Truth 处理（用于训练数据生成）"""
    print("=" * 60)
    print("示例 4: 带 Ground Truth 处理")
    print("=" * 60)
    
    config = get_default_config()
    
    # Stage1 with GT
    stage1_output = run_stage1(
        chip_id="chip_with_gt",
        raw_image_path=Path("data/chip001.png"),
        gt_image_path=Path("data/chip001_gt.png"),  # 提供 GT
        output_dir=Path("runs/with_gt/stage1"),
        config=config.stage1
    )
    
    print(f"✓ GT 切片已保存到 chamber_slices.npz (key='gt_slices')")


if __name__ == "__main__":
    import sys
    
    # 运行示例
    examples = {
        "1": ("单芯片处理", example_single_chip),
        "2": ("批量处理", example_batch_processing),
        "3": ("自定义配置", example_custom_config),
        "4": ("带GT处理", example_with_gt),
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        _, func = examples[sys.argv[1]]
        func()
    else:
        print("可用示例:")
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        print("\n使用方法: python examples/end_to_end.py [示例编号]")
        print("示例: python examples/end_to_end.py 1")
