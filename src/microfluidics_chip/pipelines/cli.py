"""
CLI 统一入口（使用 Typer）

命令：
- stage1: 处理单个图像
- stage1-batch: 批量处理
- stage2: Stage2 处理（只接受 --stage1-run-dir）
- train: 训练 UNet 模型

遵循 v1.1 规范：
- P2: stage2 只接受 stage1_run_dir 参数
- 配置文件支持（YAML）
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
import sys

# 导入核心模块
from ..core.config import (
    load_config_from_yaml,
    merge_configs,
    get_default_config,
    MicrofluidicsConfig
)
from ..core.logger import setup_logger
from ..pipelines.stage1 import run_stage1, run_stage1_batch
from ..pipelines.stage2 import run_stage2, run_stage2_batch

# 创建 Typer 应用
app = typer.Typer(
    name="microfluidics-chip",
    help="微流控芯片图像处理流水线",
    add_completion=False
)

console = Console()


# ==================== 配置加载辅助函数 ====================

def load_merged_config(config_path: Optional[Path]) -> MicrofluidicsConfig:
    """
    加载并合并配置
    
    优先级：default.yaml < config_path
    
    :param config_path: 用户指定的配置文件路径
    :return: 合并后的配置
    """
    # 1. 加载默认配置
    default_config = get_default_config()
    
    # 2. 如果用户提供了配置文件，合并
    if config_path and config_path.exists():
        user_config = load_config_from_yaml(config_path)
        return merge_configs(default_config, user_config)
    
    return default_config


# ==================== Stage1 命令 ====================

@app.command(name="stage1")
def stage1_command(
    raw_image: Path = typer.Argument(..., help="原始图像路径"),
    output_dir: Path = typer.Option("runs/stage1", "--output", "-o", help="输出目录"),
    gt_image: Optional[Path] = typer.Option(None, "--gt", help="GT图像路径（可选）"),
    chip_id: Optional[str] = typer.Option(None, "--chip-id", help="芯片ID（默认从文件名提取）"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    save_individual_slices: bool = typer.Option(False, "--save-slices", help="保存单个切片图像（调试用）"),
    save_debug: bool = typer.Option(True, "--save-debug/--no-save-debug", help="保存检测调试图像"),
    adaptive: Optional[bool] = typer.Option(
        None,
        "--adaptive/--no-adaptive",
        help="启用自适应粗到精检测 + 质量闸门（默认跟随配置文件）"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    处理单个图像（Stage1）
    
    示例：
        microfluidics-chip stage1 data/chip001.png -o runs/stage1
        microfluidics-chip stage1 data/chip001.png --gt data/chip001_gt.png
    """
    # 提取 chip_id（提前提取用于日志文件名）
    if chip_id is None:
        chip_id = raw_image.stem

    # 设置日志
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(
        level=log_level,
        log_file=output_dir / chip_id / "stage1_execution.log"
    )
    
    # 加载配置
    config = load_merged_config(config_file)
    
    console.print(f"[bold green]Processing chip: {chip_id}[/bold green]")
    
    try:
        # 运行 Stage1
        output = run_stage1(
            chip_id=chip_id,
            raw_image_path=raw_image,
            gt_image_path=gt_image,
            output_dir=output_dir,
            config=config.stage1,
            save_individual_slices=save_individual_slices,
            save_debug=save_debug,
            use_adaptive=adaptive
        )
        
        # 显示结果
        table = Table(title=f"Stage1 Output: {chip_id}")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Chip ID", output.chip_id)
        table.add_row("Chambers", str(output.num_chambers))
        table.add_row("Processing Time", f"{output.processing_time:.2f}s")
        table.add_row("Output Directory", str(output_dir / chip_id))
        
        console.print(table)
        console.print("[bold green]✓ Stage1 complete![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        sys.exit(1)


@app.command(name="stage1-batch")
def stage1_batch_command(
    input_dir: Path = typer.Argument(..., help="输入目录"),
    output_dir: Path = typer.Option("runs/stage1", "--output", "-o", help="输出目录"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    gt_suffix: str = typer.Option("_gt", "--gt-suffix", help="GT文件后缀"),
    adaptive: Optional[bool] = typer.Option(
        None,
        "--adaptive/--no-adaptive",
        help="启用自适应粗到精检测 + 质量闸门（默认跟随配置文件）"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    批量处理图像（Stage1）
    
    示例：
        microfluidics-chip stage1-batch data/raw_images -o runs/stage1
    """
    # 设置日志
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(
        level=log_level,
        log_file=output_dir / "batch_processing.log"
    )
    
    # 加载配置
    config = load_merged_config(config_file)
    
    console.print(f"[bold green]Batch processing from: {input_dir}[/bold green]")
    
    try:
        # 运行批处理
        outputs = run_stage1_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config.stage1,
            gt_suffix=gt_suffix,
            use_adaptive=adaptive
        )
        
        # 显示结果
        console.print(f"[bold green]✓ Processed {len(outputs)} chips successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        sys.exit(1)


# ==================== Stage2 命令 ====================

@app.command(name="stage2")
def stage2_command(
    stage1_run_dir: Path = typer.Argument(..., help="Stage1运行目录"),
    output_dir: Path = typer.Option("runs/stage2", "--output", "-o", help="输出目录"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    Stage2 处理（光照校正）
    
    P2 规范：只接受 stage1_run_dir 参数
    
    示例：
        microfluidics-chip stage2 runs/stage1/chip001 -o runs/stage2
    """
    chip_id = stage1_run_dir.name
    
    # 设置日志
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(
        level=log_level,
        log_file=output_dir / chip_id / "stage2_execution.log"
    )
    
    # 加载配置
    config = load_merged_config(config_file)
    
    console.print(f"[bold green]Processing chip: {chip_id}[/bold green]")
    
    try:
        # 运行 Stage2
        output = run_stage2(
            stage1_run_dir=stage1_run_dir,
            output_dir=output_dir,
            config=config.stage2
        )
        
        # 显示结果
        table = Table(title=f"Stage2 Output: {chip_id}")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Chip ID", output.chip_id)
        table.add_row("Processing Time", f"{output.processing_time:.2f}s")
        table.add_row("Output Directory", str(output_dir / chip_id))
        
        console.print(table)
        console.print("[bold green]✓ Stage2 complete![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        sys.exit(1)


@app.command(name="stage2-batch")
def stage2_batch_command(
    stage1_output_dir: Path = typer.Argument(..., help="Stage1输出根目录"),
    output_dir: Path = typer.Option("runs/stage2", "--output", "-o", help="输出目录"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    批量 Stage2 处理
    
    示例：
        microfluidics-chip stage2-batch runs/stage1 -o runs/stage2
    """
    # 设置日志
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(
        level=log_level,
        log_file=output_dir / "batch_processing.log"
    )
    
    # 加载配置
    config = load_merged_config(config_file)
    
    console.print(f"[bold green]Batch processing from: {stage1_output_dir}[/bold green]")
    
    try:
        # 运行批处理
        outputs = run_stage2_batch(
            stage1_output_dir=stage1_output_dir,
            output_dir=output_dir,
            config=config.stage2
        )
        
        # 显示结果
        console.print(f"[bold green]✓ Processed {len(outputs)} chips successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        sys.exit(1)


# ==================== 主入口 ====================

if __name__ == "__main__":
    app()
