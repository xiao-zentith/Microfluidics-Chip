"""
CLI 统一入口（使用 Typer）

命令：
- stage1: 处理单个图像
- stage1-batch: 批量处理
- stage1-yolo: 仅 YOLO 检测（无后处理）
- stage1-yolo-batch: 批量仅 YOLO 检测（无后处理）
- stage1-yolo-adaptive: 两阶段 YOLO 检测（无后处理）
- stage1-yolo-adaptive-batch: 批量两阶段 YOLO 检测（无后处理）
- stage1-post: 仅执行后处理（检测结果来自 JSON）
- stage1-post-batch: 批量仅执行后处理（检测结果来自 JSON）
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
    MicrofluidicsConfig,
    AdaptiveDetectionConfig
)
from ..core.logger import setup_logger
from ..pipelines.stage1 import (
    run_stage1,
    run_stage1_batch,
    run_stage1_yolo_only,
    run_stage1_yolo_only_batch,
    run_stage1_yolo_adaptive_only,
    run_stage1_yolo_adaptive_only_batch,
    run_stage1_postprocess_from_json,
    run_stage1_postprocess_batch,
)
from ..pipelines.stage2 import run_stage2, run_stage2_batch

# 创建 Typer 应用
app = typer.Typer(
    name="microfluidics-chip",
    help="微流控芯片图像处理流水线",
    add_completion=False
)

console = Console()


# ==================== 配置加载辅助函数 ====================

def _resolve_default_yaml_path() -> Optional[Path]:
    """
    解析默认配置文件路径，优先仓库内 `configs/default.yaml`。
    """
    candidates = [
        Path.cwd() / "configs" / "default.yaml",
        Path(__file__).resolve().parents[3] / "configs" / "default.yaml",
    ]

    for path in candidates:
        if path.exists():
            return path
    return None


def load_merged_config(config_path: Optional[Path]) -> MicrofluidicsConfig:
    """
    加载并合并配置
    
    优先级：default.yaml < config_path
    
    :param config_path: 用户指定的配置文件路径
    :return: 合并后的配置
    """
    # 1. 优先加载 configs/default.yaml，找不到时回退内置默认配置
    default_yaml = _resolve_default_yaml_path()
    if default_yaml is not None:
        default_config = load_config_from_yaml(default_yaml)
    else:
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


@app.command(name="stage1-yolo")
def stage1_yolo_command(
    raw_image: Path = typer.Argument(..., help="原始图像路径"),
    output_dir: Path = typer.Option("runs/stage1_yolo", "--output", "-o", help="输出目录"),
    chip_id: Optional[str] = typer.Option(None, "--chip-id", help="芯片ID（默认从文件名提取）"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    confidence: Optional[float] = typer.Option(None, "--conf", min=0.0, max=1.0, help="YOLO置信度阈值覆盖"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    仅执行 YOLO 检测（不进行 Stage1 后处理），并保存可视化结果。
    """
    if chip_id is None:
        chip_id = raw_image.stem

    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level, log_file=output_dir / chip_id / "stage1_yolo_execution.log")

    config = load_merged_config(config_file)
    if confidence is not None:
        config.stage1.yolo.confidence_threshold = confidence

    try:
        payload = run_stage1_yolo_only(
            chip_id=chip_id,
            raw_image_path=raw_image,
            output_dir=output_dir,
            config=config.stage1,
        )

        table = Table(title=f"Stage1 YOLO-only Output: {chip_id}")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Chip ID", payload["chip_id"])
        table.add_row("Detections", str(payload["num_detections"]))
        table.add_row("Processing Time", f"{payload['processing_time']:.2f}s")
        table.add_row("Output Directory", str(output_dir / chip_id))
        console.print(table)
        console.print("[bold green]✓ Stage1 YOLO-only complete![/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        sys.exit(1)


@app.command(name="stage1-yolo-batch")
def stage1_yolo_batch_command(
    input_dir: Path = typer.Argument(..., help="输入目录"),
    output_dir: Path = typer.Option("runs/stage1_yolo", "--output", "-o", help="输出目录"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    confidence: Optional[float] = typer.Option(None, "--conf", min=0.0, max=1.0, help="YOLO置信度阈值覆盖"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    批量仅执行 YOLO 检测（不进行 Stage1 后处理）。
    """
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level, log_file=output_dir / "stage1_yolo_batch.log")

    config = load_merged_config(config_file)
    if confidence is not None:
        config.stage1.yolo.confidence_threshold = confidence

    try:
        outputs = run_stage1_yolo_only_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config.stage1,
        )
        console.print(f"[bold green]✓ YOLO-only batch processed {len(outputs)} images![/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        sys.exit(1)


@app.command(name="stage1-yolo-adaptive")
def stage1_yolo_adaptive_command(
    raw_image: Path = typer.Argument(..., help="原始图像路径"),
    output_dir: Path = typer.Option("runs/stage1_yolo_adaptive", "--output", "-o", help="输出目录"),
    chip_id: Optional[str] = typer.Option(None, "--chip-id", help="芯片ID（默认从文件名提取）"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    coarse_conf: Optional[float] = typer.Option(None, "--coarse-conf", min=0.0, max=1.0, help="粗扫描置信度覆盖"),
    fine_conf: Optional[float] = typer.Option(None, "--fine-conf", min=0.0, max=1.0, help="精扫描置信度覆盖"),
    fine_imgsz: Optional[int] = typer.Option(None, "--fine-imgsz", min=32, help="精扫描分辨率覆盖"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    仅执行两阶段 YOLO 检测（粗到精 + ROI 聚类），不进行拓扑/几何后处理。
    """
    if chip_id is None:
        chip_id = raw_image.stem

    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level, log_file=output_dir / chip_id / "stage1_yolo_adaptive_execution.log")

    config = load_merged_config(config_file)
    if any(v is not None for v in (coarse_conf, fine_conf, fine_imgsz)):
        if config.stage1.adaptive_detection is None:
            config.stage1.adaptive_detection = AdaptiveDetectionConfig()
        if coarse_conf is not None:
            config.stage1.adaptive_detection.coarse_conf = coarse_conf
        if fine_conf is not None:
            config.stage1.adaptive_detection.fine_conf = fine_conf
        if fine_imgsz is not None:
            config.stage1.adaptive_detection.fine_imgsz = fine_imgsz

    try:
        payload = run_stage1_yolo_adaptive_only(
            chip_id=chip_id,
            raw_image_path=raw_image,
            output_dir=output_dir,
            config=config.stage1,
        )

        table = Table(title=f"Stage1 Adaptive YOLO-only Output: {chip_id}")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Chip ID", payload["chip_id"])
        table.add_row("Detections", str(payload["num_detections"]))
        table.add_row("Cluster Score", f"{payload['cluster_score']:.3f}")
        table.add_row("Fallback Cluster", str(payload["is_fallback"]))
        table.add_row("Processing Time", f"{payload['processing_time']:.2f}s")
        table.add_row("Output Directory", str(output_dir / chip_id))
        console.print(table)
        console.print("[bold green]✓ Stage1 adaptive YOLO-only complete![/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        sys.exit(1)


@app.command(name="stage1-yolo-adaptive-batch")
def stage1_yolo_adaptive_batch_command(
    input_dir: Path = typer.Argument(..., help="输入目录"),
    output_dir: Path = typer.Option("runs/stage1_yolo_adaptive", "--output", "-o", help="输出目录"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    coarse_conf: Optional[float] = typer.Option(None, "--coarse-conf", min=0.0, max=1.0, help="粗扫描置信度覆盖"),
    fine_conf: Optional[float] = typer.Option(None, "--fine-conf", min=0.0, max=1.0, help="精扫描置信度覆盖"),
    fine_imgsz: Optional[int] = typer.Option(None, "--fine-imgsz", min=32, help="精扫描分辨率覆盖"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    批量仅执行两阶段 YOLO 检测（不进行拓扑/几何后处理）。
    """
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level, log_file=output_dir / "stage1_yolo_adaptive_batch.log")

    config = load_merged_config(config_file)
    if any(v is not None for v in (coarse_conf, fine_conf, fine_imgsz)):
        if config.stage1.adaptive_detection is None:
            config.stage1.adaptive_detection = AdaptiveDetectionConfig()
        if coarse_conf is not None:
            config.stage1.adaptive_detection.coarse_conf = coarse_conf
        if fine_conf is not None:
            config.stage1.adaptive_detection.fine_conf = fine_conf
        if fine_imgsz is not None:
            config.stage1.adaptive_detection.fine_imgsz = fine_imgsz

    try:
        outputs = run_stage1_yolo_adaptive_only_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config.stage1,
        )
        console.print(f"[bold green]✓ Adaptive YOLO-only batch processed {len(outputs)} images![/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        sys.exit(1)


@app.command(name="stage1-post")
def stage1_post_command(
    detections_json: Path = typer.Argument(..., help="检测结果 JSON 路径"),
    output_dir: Path = typer.Option("runs/stage1_post", "--output", "-o", help="输出目录"),
    raw_image: Optional[Path] = typer.Option(None, "--raw-image", help="原图路径覆盖（默认读取 JSON 中 raw_image_path）"),
    chip_id: Optional[str] = typer.Option(None, "--chip-id", help="芯片ID覆盖"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    min_topology_detections: Optional[int] = typer.Option(
        None,
        "--min-topology-detections",
        min=2,
        max=12,
        help="拓扑拟合最小检测点数阈值（默认跟随配置 min_detections）"
    ),
    fallback_detection: bool = typer.Option(
        True,
        "--fallback-detection/--no-fallback-detection",
        help="拓扑拟合失败时是否使用宽松两阶段检测重试"
    ),
    export_geo_even_if_blank_unresolved: bool = typer.Option(
        True,
        "--export-geo-even-if-blank-unresolved/--strict-semantic-success",
        help="几何通过即导出切片；关闭时要求语义(Blank/ReferenceArm)也通过",
    ),
    save_individual_slices: bool = typer.Option(False, "--save-slices", help="保存单个切片图像（调试用）"),
    save_debug: bool = typer.Option(True, "--save-debug/--no-save-debug", help="保存检测调试图像"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    仅执行 Stage1 后处理（几何校正 + 切片），检测结果由 JSON 提供。
    """
    effective_chip_id = chip_id or detections_json.parent.name
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level, log_file=output_dir / effective_chip_id / "stage1_post_execution.log")

    config = load_merged_config(config_file)

    try:
        output = run_stage1_postprocess_from_json(
            detections_json_path=detections_json,
            output_dir=output_dir,
            config=config.stage1,
            raw_image_path=raw_image,
            chip_id=chip_id,
            min_topology_detections=min_topology_detections,
            enable_fallback_detection=fallback_detection,
            export_geo_even_if_blank_unresolved=export_geo_even_if_blank_unresolved,
            save_individual_slices=save_individual_slices,
            save_debug=save_debug
        )

        table = Table(title=f"Stage1 Postprocess-only Output: {output.chip_id}")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Chip ID", output.chip_id)
        table.add_row("Chambers", str(output.num_chambers))
        table.add_row("Processing Time", f"{output.processing_time:.2f}s")
        table.add_row("Output Directory", str(output_dir / output.chip_id))
        console.print(table)
        console.print("[bold green]✓ Stage1 postprocess-only complete![/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        sys.exit(1)


@app.command(name="stage1-post-batch")
def stage1_post_batch_command(
    input_dir: Path = typer.Argument(..., help="输入目录（递归查找检测 JSON）"),
    output_dir: Path = typer.Option("runs/stage1_post", "--output", "-o", help="输出目录"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    json_name: Optional[str] = typer.Option(None, "--json-name", help="仅处理指定 JSON 文件名"),
    min_topology_detections: Optional[int] = typer.Option(
        None,
        "--min-topology-detections",
        min=2,
        max=12,
        help="拓扑拟合最小检测点数阈值（默认跟随配置 min_detections）"
    ),
    fallback_detection: bool = typer.Option(
        True,
        "--fallback-detection/--no-fallback-detection",
        help="拓扑拟合失败时是否使用宽松两阶段检测重试"
    ),
    export_geo_even_if_blank_unresolved: bool = typer.Option(
        True,
        "--export-geo-even-if-blank-unresolved/--strict-semantic-success",
        help="几何通过即导出切片；关闭时要求语义(Blank/ReferenceArm)也通过",
    ),
    save_individual_slices: bool = typer.Option(False, "--save-slices", help="保存单个切片图像（调试用）"),
    save_debug: bool = typer.Option(True, "--save-debug/--no-save-debug", help="保存检测调试图像"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志")
):
    """
    批量仅执行 Stage1 后处理（几何校正 + 切片），检测结果由 JSON 提供。
    """
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level, log_file=output_dir / "stage1_post_batch.log")

    config = load_merged_config(config_file)

    try:
        outputs = run_stage1_postprocess_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config.stage1,
            json_name=json_name,
            min_topology_detections=min_topology_detections,
            enable_fallback_detection=fallback_detection,
            export_geo_even_if_blank_unresolved=export_geo_even_if_blank_unresolved,
            save_individual_slices=save_individual_slices,
            save_debug=save_debug
        )
        console.print(f"[bold green]✓ Stage1 postprocess-only batch processed {len(outputs)} chips![/bold green]")
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
