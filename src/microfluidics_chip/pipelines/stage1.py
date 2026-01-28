"""
Stage1 业务编排层（IO + Inference）
遵循 v1.1 强制规范：
- P2: 固定文件命名
- P4: 批处理循环外初始化
"""

import cv2
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
from ..core.types import Stage1Output
from ..core.config import Stage1Config
from ..core.io import save_stage1_result
from ..core.logger import get_logger
from ..stage1_detection.detector import ChamberDetector
from ..stage1_detection.geometry_engine import CrossGeometryEngine
from ..stage1_detection.inference import infer_stage1

logger = get_logger("pipelines.stage1")


def run_stage1(
    chip_id: str,
    raw_image_path: Path,
    gt_image_path: Optional[Path],
    output_dir: Path,
    config: Stage1Config,
    detector: Optional[ChamberDetector] = None,
    geometry_engine: Optional[CrossGeometryEngine] = None,
    save_individual_slices: bool = False,
    save_debug: bool = True
) -> Stage1Output:
    """
    运行单个芯片的 Stage1 处理
    
    流程：
    1. 读取图像
    2. 调用 infer_stage1() 推理
    3. P2: 使用 save_stage1_result() 保存（固定文件名）
    
    :param chip_id: 芯片ID
    :param raw_image_path: 原始图像路径
    :param gt_image_path: GT图像路径（可选）
    :param output_dir: 输出目录（会在此目录下创建 chip_id 子目录）
    :param config: Stage1 配置
    :param detector: 检测器实例（批处理时复用）
    :param geometry_engine: 几何引擎实例（批处理时复用）
    :param save_individual_slices: 是否保存单个切片图像（用于调试）
    :param save_debug: 是否保存调试可视化图像（检测框、类别标注等）
    :return: Stage1Output（落盘结果）
    """
    logger.info(f"[{chip_id}] Starting Stage1 processing...")
    
    # ==================== 读取图像 ====================
    raw_image = cv2.imread(str(raw_image_path))
    if raw_image is None:
        raise FileNotFoundError(f"Cannot read raw image: {raw_image_path}")
    
    gt_image = None
    if gt_image_path and gt_image_path.exists():
        gt_image = cv2.imread(str(gt_image_path))
        if gt_image is None:
            logger.warning(f"[{chip_id}] Cannot read GT image: {gt_image_path}")
    
    # ==================== 推理 ====================
    result = infer_stage1(
        chip_id=chip_id,
        raw_image=raw_image,
        gt_image=gt_image,
        config=config,
        detector=detector,
        geometry_engine=geometry_engine
    )
    
    # ==================== P2: 保存结果（固定文件名） ====================
    run_dir = output_dir / chip_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    save_gt = (result.gt_slices is not None)
    output = save_stage1_result(
        result, 
        run_dir, 
        save_gt=save_gt,
        save_individual_slices=save_individual_slices
    )
    
    # 保存调试可视化（如果有）
    if save_debug and result.debug_vis is not None:
        debug_path = run_dir / "debug_detection.png"
        cv2.imwrite(str(debug_path), result.debug_vis)
        logger.info(f"[{chip_id}] Debug visualization saved: debug_detection.png")
    
    logger.info(f"[{chip_id}] Stage1 output saved to: {run_dir}")
    logger.info(f"[{chip_id}] Files: stage1_metadata.json, aligned.png, chamber_slices.npz")
    
    return output


def run_stage1_batch(
    input_dir: Path,
    output_dir: Path,
    config: Stage1Config,
    gt_suffix: str = "_gt",
    image_extensions: List[str] = [".png", ".jpg", ".jpeg"]
) -> List[Stage1Output]:
    """
    批量运行 Stage1
    
    P4 强制规范：
    - 循环外初始化 detector 和 geometry_engine
    - 循环内复用实例（避免重复加载模型）
    
    目录结构假设：
    input_dir/
      ├── chip001.png
      ├── chip001_gt.png
      ├── chip002.png
      └── chip002_gt.png
    
    :param input_dir: 输入目录
    :param output_dir: 输出根目录
    :param config: Stage1 配置
    :param gt_suffix: GT 文件后缀（如 "_gt"）
    :param image_extensions: 支持的图像扩展名
    :return: Stage1Output 列表
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 收集文件对 ====================
    file_pairs = []
    
    for ext in image_extensions:
        for raw_path in input_dir.glob(f"*{ext}"):
            # 跳过 GT 文件
            if gt_suffix in raw_path.stem:
                continue
            
            chip_id = raw_path.stem
            
            # 查找对应的 GT 文件
            gt_path = raw_path.parent / f"{chip_id}{gt_suffix}{ext}"
            if not gt_path.exists():
                gt_path = None
            
            file_pairs.append((chip_id, raw_path, gt_path))
    
    if not file_pairs:
        logger.warning(f"No image files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(file_pairs)} image pairs")
    
    # ==================== P4: 循环外初始化模型 ====================
    logger.info("Initializing detector and geometry engine (P4 batch optimization)...")
    detector = ChamberDetector(config.yolo)
    geometry_engine = CrossGeometryEngine(config.geometry)
    logger.info("Models initialized successfully")
    
    # ==================== 批处理循环 ====================
    outputs = []
    
    for idx, (chip_id, raw_path, gt_path) in enumerate(tqdm(file_pairs, desc="Processing chips")):
        try:
            output = run_stage1(
                chip_id=chip_id,
                raw_image_path=raw_path,
                gt_image_path=gt_path,
                output_dir=output_dir,
                config=config,
                detector=detector,           # P4: 复用实例
                geometry_engine=geometry_engine  # P4: 复用实例
            )
            outputs.append(output)
            
            # 进度提示
            logger.info(f"✓ {chip_id} completed ({idx+1}/{len(file_pairs)})")
            
        except Exception as e:
            logger.error(f"✗ {chip_id} failed: {e}")
            continue
    
    # ==================== 汇总 ====================
    success_count = len(outputs)
    fail_count = len(file_pairs) - success_count
    
    logger.info(f"Batch processing complete: {success_count} success, {fail_count} failed")
    
    return outputs
