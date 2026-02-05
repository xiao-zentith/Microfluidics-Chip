"""
Stage1 统一预处理模块

提供训练/推理一致的预处理函数，确保增强策略在离线增强和在线推理中保持同步。

功能：
- CLAHE 对比度增强 (LAB L通道)
- Invert 亮度反转
- 组合预处理流水线

设计原则：
- 所有函数可复用于离线增强脚本和在线推理
- 参数可配置，便于训练/推理一致性调试
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    对图像应用 CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    仅对 LAB 色彩空间的 L 通道进行增强，保持色彩不变。
    
    :param image: BGR 图像 (H, W, 3) uint8
    :param clip_limit: 对比度限制阈值 (默认 2.0)
    :param tile_grid_size: CLAHE 网格大小 (默认 8x8)
    :return: CLAHE 增强后的图像
    
    示例:
        >>> enhanced = apply_clahe(image, clip_limit=3.0)
    """
    if image is None or image.size == 0:
        return image
    
    # 转换到 LAB 色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 创建 CLAHE 对象并应用到 L 通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    # 合并通道并转回 BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_invert(image: np.ndarray) -> np.ndarray:
    """
    反转图像亮度
    
    用于处理暗背景亮目标的场景，或作为数据增强手段。
    
    :param image: 输入图像 (H, W, 3) uint8
    :return: 反转后的图像
    """
    if image is None or image.size == 0:
        return image
    
    return 255 - image


def preprocess_image(
    image: np.ndarray,
    enable_clahe: bool = True,
    enable_invert: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    统一预处理流水线
    
    应用顺序：CLAHE -> Invert (如果启用)
    
    :param image: 输入图像 (H, W, 3) uint8
    :param enable_clahe: 是否启用 CLAHE (默认 True)
    :param enable_invert: 是否启用 Invert (默认 False)
    :param clahe_clip_limit: CLAHE clip limit
    :param clahe_tile_size: CLAHE 网格大小
    :return: 预处理后的图像
    
    重要：
    确保训练数据增强与推理预处理使用相同的参数！
    
    示例:
        # 训练时
        aug_img = preprocess_image(img, enable_clahe=True, enable_invert=False)
        
        # 推理时 (必须与训练一致)
        inf_img = preprocess_image(img, enable_clahe=True, enable_invert=False)
    """
    if image is None or image.size == 0:
        return image
    
    result = image.copy()
    
    if enable_clahe:
        result = apply_clahe(result, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_size)
    
    if enable_invert:
        result = apply_invert(result)
    
    return result


def extract_roi(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 0
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    从图像中提取 ROI 区域
    
    :param image: 输入图像 (H, W, 3)
    :param bbox: ROI 边界框 (x1, y1, x2, y2)
    :param padding: 额外填充像素
    :return: (roi_image, actual_bbox) - ROI 图像和实际裁剪的边界框
    
    说明：
    - 会自动处理边界情况
    - actual_bbox 返回实际裁剪的坐标（可能因边界而调整）
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # 应用 padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    actual_bbox = (x1, y1, x2, y2)
    roi = image[y1:y2, x1:x2].copy()
    
    return roi, actual_bbox


def resize_for_detection(
    image: np.ndarray,
    target_size: int,
    keep_aspect: bool = True
) -> Tuple[np.ndarray, float, float]:
    """
    为检测模型调整图像大小
    
    :param image: 输入图像
    :param target_size: 目标尺寸 (正方形边长)
    :param keep_aspect: 是否保持宽高比
    :return: (resized_image, scale_x, scale_y) - 调整后的图像和缩放比例
    
    说明：
    scale_x, scale_y 用于将检测结果映射回原图坐标
    """
    h, w = image.shape[:2]
    
    if keep_aspect:
        # 按较长边缩放
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建正方形画布并居中放置
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # 返回缩放比例 (需要考虑偏移)
        scale_x = w / new_w
        scale_y = h / new_h
        return canvas, scale_x, scale_y
    else:
        # 直接拉伸
        resized = cv2.resize(image, (target_size, target_size))
        scale_x = w / target_size
        scale_y = h / target_size
        return resized, scale_x, scale_y
