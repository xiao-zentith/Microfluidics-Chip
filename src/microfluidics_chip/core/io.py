"""
统一 IO 管理器
职责：保存/加载 Result ↔ Output
遵循 v1.1 强制规范：
- P1: 相对路径（str）→ 绝对路径（Path）显式转换
- P2: 固定文件命名，禁止Glob
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2
import json
from .types import Stage1Result, Stage1Output, Stage2Result, Stage2Output


class ResultSaver:
    """
    Result 保存器（窄职责）
    ⚠️ P2规范：禁止自动加前缀/时间戳/chip_id
    文件名必须由调用者指定（通常为固定名称）
    """
    
    def __init__(self, run_dir: Path):
        """
        :param run_dir: 保存目录（由调用者决定结构）
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def save_npz(
        self,
        filename: str,
        data: Dict[str, np.ndarray],
        compressed: bool = True
    ) -> Path:
        """
        保存 npz 文件
        
        :param filename: 固定文件名（如 "chamber_slices.npz"）
        :param data: key-value 字典（如 {"slices": array}）
        :param compressed: 是否压缩
        :return: 保存的绝对路径
        """
        path = self.run_dir / filename
        if compressed:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)
        return path
    
    def load_npz(self, filename: str, key: str) -> np.ndarray:
        """
        加载 npz 文件的指定 key
        
        :param filename: 固定文件名
        :param key: npz 内部 key（P2规范统一为 'slices'）
        :return: numpy数组
        """
        path = self.run_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"npz文件不存在: {path}")
        
        data = np.load(path)
        if key not in data:
            raise KeyError(f"npz文件缺少key '{key}': {path}")
        
        return data[key]
    
    def save_image(self, filename: str, image: np.ndarray) -> Path:
        """
        保存图像
        
        :param filename: 固定文件名（如 "aligned.png"）
        :param image: 图像数组 (H, W, 3)
        :return: 保存的绝对路径
        """
        path = self.run_dir / filename
        success = cv2.imwrite(str(path), image)
        if not success:
            raise IOError(f"图像保存失败: {path}")
        return path
    
    def load_image(self, filename: str) -> np.ndarray:
        """加载图像"""
        path = self.run_dir / filename
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"图像文件不存在或无法读取: {path}")
        return img
    
    def save_json(self, filename: str, data: Dict[str, Any]) -> Path:
        """
        保存 JSON
        
        :param filename: 固定文件名（如 "stage1_metadata.json"）
        :param data: 字典数据
        :return: 保存的绝对路径
        """
        path = self.run_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path
    
    def load_json(self, filename: str) -> Dict[str, Any]:
        """加载 JSON"""
        path = self.run_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"JSON文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


# ==================== Stage1 专用保存函数 ====================

def save_stage1_result(
    result: Stage1Result,
    run_dir: Path,
    save_gt: bool = False,
    save_individual_slices: bool = False
) -> Stage1Output:
    """
    保存 Stage1Result → 返回 Stage1Output
    
    P2规范：固定文件名
    - stage1_metadata.json
    - aligned.png
    - chamber_slices.npz (key="slices", 可选 key="gt_slices")
    - debug_visualization.png (可选)
    - slices/ (可选，单个切片目录)
    
    :param result: Stage1内存结果
    :param run_dir: 保存目录
    :param save_gt: 是否保存GT切片
    :param save_individual_slices: 是否保存单个切片图像（用于调试）
    :return: Stage1Output（落盘结果）
    """
    saver = ResultSaver(run_dir)
    
    # 1. 保存图像
    saver.save_image("aligned.png", result.aligned_image)
    
    # 2. P2规范：保存npz（统一key命名）
    npz_data = {"slices": result.chamber_slices}
    if save_gt and result.gt_slices is not None:
        npz_data["gt_slices"] = result.gt_slices
    saver.save_npz("chamber_slices.npz", npz_data)
    
    # 3. 保存调试可视化（可选）
    if result.debug_vis is not None:
        saver.save_image("debug_visualization.png", result.debug_vis)
    
    # 4. 保存单个切片（可选，用于调试）
    if save_individual_slices:
        slices_dir = run_dir / "slices"
        slices_dir.mkdir(exist_ok=True)
        
        for i in range(len(result.chamber_slices)):
            # 保存 Raw 切片
            raw_slice_path = slices_dir / f"{i}_raw.jpg"
            cv2.imwrite(str(raw_slice_path), result.chamber_slices[i])
            
            # 保存 GT 切片（如果有）
            if save_gt and result.gt_slices is not None and i < len(result.gt_slices):
                gt_slice_path = slices_dir / f"{i}_gt.jpg"
                cv2.imwrite(str(gt_slice_path), result.gt_slices[i])
    
    # 5. 转换为 Output
    from .types import stage1_result_to_output
    output = stage1_result_to_output(result, run_dir)
    
    # 6. P2规范：保存metadata（固定文件名）
    saver.save_json("stage1_metadata.json", output.model_dump())
    
    return output


def load_stage1_output(run_dir: Path) -> Tuple[Stage1Output, np.ndarray]:
    """
    从固定文件名加载 Stage1 产物（P2/P1规范）
    
    P2规范：只认固定文件名，禁止Glob
    P1规范：相对路径（str）→ 绝对路径（Path）显式转换
    
    :param run_dir: Stage1运行目录
    :return: (Stage1Output, chamber_slices: np.ndarray)
    """
    saver = ResultSaver(run_dir)
    
    # P2: 硬编码固定文件名（禁止Glob）
    metadata_path = run_dir / "stage1_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"未找到 stage1_metadata.json，请检查目录是否为有效的Stage1输出: {run_dir}"
        )
    
    # 1. 加载 metadata
    metadata = saver.load_json("stage1_metadata.json")
    output = Stage1Output(**metadata)
    
    # P1: 相对路径（str）→ 绝对路径（Path）显式转换
    slices_rel_path = output.chamber_slices_path  # str: "chamber_slices.npz"
    slices_abs_path = run_dir / Path(slices_rel_path)
    
    if not slices_abs_path.exists():
        raise FileNotFoundError(f"未找到切片文件: {slices_abs_path}")
    
    # 2. P2: 加载npz（固定key='slices'）
    slices_data = np.load(slices_abs_path)
    
    if 'slices' not in slices_data:
        raise KeyError(
            f"npz文件缺少 'slices' key（P2规范要求）: {slices_abs_path}\n"
            f"实际包含的keys: {list(slices_data.keys())}"
        )
    
    chamber_slices = slices_data['slices']
    
    return output, chamber_slices


# ==================== Stage2 专用保存函数 ====================

def save_stage2_result(
    result: Stage2Result,
    run_dir: Path
) -> Stage2Output:
    """
    保存 Stage2Result → 返回 Stage2Output
    
    P2规范：固定文件名
    - stage2_metadata.json
    - corrected_slices.npz (key="slices")
    
    :param result: Stage2内存结果
    :param run_dir: 保存目录
    :return: Stage2Output（落盘结果）
    """
    saver = ResultSaver(run_dir)
    
    # 1. P2规范：保存npz（统一key="slices"）
    saver.save_npz("corrected_slices.npz", {"slices": result.corrected_slices})
    
    # 2. 转换为 Output
    from .types import stage2_result_to_output
    output = stage2_result_to_output(result, run_dir)
    
    # 3. P2规范：保存metadata（固定文件名）
    saver.save_json("stage2_metadata.json", output.model_dump())
    
    return output


def load_stage2_output(run_dir: Path) -> Tuple[Stage2Output, np.ndarray]:
    """
    从固定文件名加载 Stage2 产物（P2/P1规范）
    
    :param run_dir: Stage2运行目录
    :return: (Stage2Output, corrected_slices: np.ndarray)
    """
    saver = ResultSaver(run_dir)
    
    # P2: 固定文件名
    metadata_path = run_dir / "stage2_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"未找到 stage2_metadata.json: {run_dir}"
        )
    
    # 1. 加载 metadata
    metadata = saver.load_json("stage2_metadata.json")
    output = Stage2Output(**metadata)
    
    # P1: 相对路径→绝对路径
    slices_abs_path = run_dir / Path(output.corrected_slices_path)
    
    # 2. P2: 加载npz（固定key）
    slices_data = np.load(slices_abs_path)
    
    if 'slices' not in slices_data:
        raise KeyError(f"npz文件缺少 'slices' key: {slices_abs_path}")
    
    corrected_slices = slices_data['slices']
    
    return output, corrected_slices
