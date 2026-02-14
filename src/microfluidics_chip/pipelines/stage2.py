"""
Stage2 业务编排层（IO + Inference）
遵循 v1.1 强制规范：
- P2: 只接受 stage1_run_dir 参数（强制链路）
- P2: 固定文件命名
"""

from pathlib import Path
from typing import Optional
from tqdm import tqdm
from ..core.types import Stage2Output
from ..core.config import Stage2Config
from ..core.io import load_stage1_output, save_stage2_result
from ..core.logger import get_logger
from ..stage2_correction.models.dual_stream_unet import RefGuidedUNet
from ..stage2_correction.inference import infer_stage2

logger = get_logger("pipelines.stage2")


def run_stage2(
    stage1_run_dir: Path,
    output_dir: Path,
    config: Stage2Config,
    model: Optional[RefGuidedUNet] = None
) -> Stage2Output:
    """
    运行单个芯片的 Stage2 处理
    
    P2 强制规范：
    - 只接受 stage1_run_dir 参数
    - 从固定文件名加载 Stage1 产物
    - 保存时使用固定文件名
    
    流程：
    1. P2: 从 stage1_run_dir 加载切片（固定文件名）
    2. 调用 infer_stage2() 推理
    3. P2: 保存结果（固定文件名）
    
    :param stage1_run_dir: Stage1 运行目录（如 runs/stage1/chip001）
    :param output_dir: Stage2 输出根目录
    :param config: Stage2 配置
    :param model: UNet 模型实例（批处理时复用）
    :return: Stage2Output（落盘结果）
    """
    stage1_run_dir = Path(stage1_run_dir)
    
    # P2: 从 Stage1 产物中提取 chip_id
    # 假设目录名就是 chip_id（如 runs/stage1/chip001）
    chip_id = stage1_run_dir.name
    
    logger.info(f"[{chip_id}] Starting Stage2 processing...")
    logger.info(f"[{chip_id}] Loading Stage1 output from: {stage1_run_dir}")
    
    # ==================== P2: 加载 Stage1 产物（固定文件名） ====================
    stage1_output, chamber_slices = load_stage1_output(stage1_run_dir)
    
    logger.info(f"[{chip_id}] Loaded {len(chamber_slices)} slices from Stage1")
    
    # ==================== 推理 ====================
    result = infer_stage2(
        chip_id=chip_id,
        chamber_slices=chamber_slices,
        config=config,
        model=model
    )
    
    # ==================== P2: 保存结果（固定文件名） ====================
    run_dir = output_dir / chip_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    output = save_stage2_result(result, run_dir)
    
    logger.info(f"[{chip_id}] Stage2 output saved to: {run_dir}")
    logger.info(f"[{chip_id}] Files: stage2_metadata.json, corrected_slices.npz")
    
    return output


def run_stage2_batch(
    stage1_output_dir: Path,
    output_dir: Path,
    config: Stage2Config
) -> list[Stage2Output]:
    """
    批量运行 Stage2
    
    P2 强制规范：
    - 遍历 stage1_output_dir 下的所有子目录
    - 每个子目录都是一个 Stage1 运行目录
    
    批处理优化：
    - 循环外初始化 UNet 模型（类似 P4）
    - 循环内复用模型实例
    
    :param stage1_output_dir: Stage1 输出根目录（如 runs/stage1）
    :param output_dir: Stage2 输出根目录（如 runs/stage2）
    :param config: Stage2 配置
    :return: Stage2Output 列表
    """
    stage1_output_dir = Path(stage1_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 收集 Stage1 运行目录 ====================
    stage1_dirs = [d for d in stage1_output_dir.iterdir() if d.is_dir()]
    
    if not stage1_dirs:
        logger.warning(f"No Stage1 output directories found in {stage1_output_dir}")
        return []
    
    logger.info(f"Found {len(stage1_dirs)} Stage1 output directories")
    
    # ==================== 批处理优化：循环外初始化模型 ====================
    import torch
    device = torch.device(config.model.device)
    
    logger.info("Initializing UNet model for batch processing...")
    model = RefGuidedUNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        features=config.model.features
    ).to(device)
    
    # 加载权重
    if config.weights_path:
        weights_path = Path(config.weights_path)
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=device))
            logger.info(f"Loaded weights from {weights_path}")
        else:
            logger.warning(f"Weights file not found: {weights_path}")
    
    model.eval()
    logger.info("Model initialized successfully")
    
    # ==================== 批处理循环 ====================
    outputs = []
    
    for idx, stage1_dir in enumerate(tqdm(stage1_dirs, desc="Processing chips")):
        try:
            output = run_stage2(
                stage1_run_dir=stage1_dir,
                output_dir=output_dir,
                config=config,
                model=model  # 复用模型实例
            )
            outputs.append(output)
            
            # 进度提示
            chip_id = stage1_dir.name
            logger.info(f"✓ {chip_id} completed ({idx+1}/{len(stage1_dirs)})")
            
        except Exception as e:
            chip_id = stage1_dir.name
            logger.error(f"✗ {chip_id} failed: {e}")
            continue
    
    # ==================== 汇总 ====================
    success_count = len(outputs)
    fail_count = len(stage1_dirs) - success_count
    
    logger.info(f"Batch processing complete: {success_count} success, {fail_count} failed")
    
    return outputs
