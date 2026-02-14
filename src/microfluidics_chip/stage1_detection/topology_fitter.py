"""
拓扑约束模板拟合器

实现 RANSAC 模板拟合 + 缺失腔室回填，确保即使暗腔室漏检也能获得完整的 12 个腔室坐标。

核心功能：
1. 十字模板定义 (可配置/可从文件加载)
2. RANSAC Similarity Transform 拟合 (旋转+缩放+平移)
3. 缺失腔室坐标回填
4. 可见性判定 (边界检查)
5. 暗腔室判定 (基于拓扑位置的亮度分析)

设计原则：
- 模板参数可配置，支持不同芯片布局
- 暗腔室判定基于拓扑位置，而非全局排序
- 提供完整的拟合质量指标
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from ..core.logger import get_logger

logger = get_logger("stage1_detection.topology_fitter")


# ==================== 默认十字模板定义 ====================

# 十字布局模板 (归一化坐标，中心为原点)
#
# 拓扑结构说明：
# - 总共 4 条臂，每条臂有 3 个腔室，无中心腔室
# - 每个臂从内到外排列：索引 0/3/6/9 为最内侧，索引 2/5/8/11 为最外侧
# - 暗腔室（唯一1个）位于其所处臂的最外侧位置
# - 由于芯片可能存在旋转，无法预知暗腔室在哪条臂，只知道是臂的最外侧
#
# 索引布局:
#           0
#           1
#           2 (最外侧, 可能是暗腔室)
#   9 10 11     3 4 5
#           6
#           7
#           8 (最外侧, 可能是暗腔室)

DEFAULT_CROSS_TEMPLATE = np.array([
    # Top arm (向上, y负方向): 内→外
    [0, -1], [0, -2], [0, -3],
    # Right arm (向右, x正方向): 内→外
    [1, 0], [2, 0], [3, 0],
    # Bottom arm (向下, y正方向): 内→外
    [0, 1], [0, 2], [0, 3],
    # Left arm (向左, x负方向): 内→外
    [-1, 0], [-2, 0], [-3, 0],
], dtype=np.float32)

# 腔室索引到臂的映射
ARM_INDICES = {
    'top': [0, 1, 2],      # index 2 是最外侧
    'right': [3, 4, 5],    # index 5 是最外侧
    'bottom': [6, 7, 8],   # index 8 是最外侧
    'left': [9, 10, 11]    # index 11 是最外侧
}

# 各臂最外侧腔室索引（暗腔室只可能出现在这些位置）
OUTERMOST_CHAMBER_INDICES = [2, 5, 8, 11]


@dataclass
class TopologyConfig:
    """拓扑拟合配置"""
    # 模板参数
    template_scale: float = 50.0  # 模板点间距 (像素)
    template_path: Optional[str] = None  # 自定义模板文件路径 (JSON)
    
    # RANSAC 参数
    ransac_iters: int = 200  # RANSAC 迭代次数
    ransac_threshold: float = 25.0  # 内点阈值 (像素)
    min_inliers: int = 4  # 最少内点数
    
    # 可见性判定
    visibility_margin: int = 10  # 边界安全距离 (像素)
    
    # 暗腔室判定
    brightness_roi_size: int = 30  # 亮度判定 ROI 尺寸
    dark_percentile: float = 25.0  # 暗腔室判定分位数
    
    # 回退参数
    fallback_to_affine: bool = True  # Similarity 失败时是否回退到 Affine


@dataclass
class FittingResult:
    """拟合结果"""
    # 核心输出
    fitted_centers: np.ndarray  # (12, 2) 所有腔室的预测坐标
    
    # 状态标记
    visibility: np.ndarray  # (12,) bool, 是否在图像内可见
    detected_mask: np.ndarray  # (12,) bool, 是否被直接检测到 (vs 回填)
    
    # 拟合信息
    transform_matrix: np.ndarray  # (2, 3) Similarity/Affine 变换矩阵
    inlier_ratio: float  # RANSAC 内点比例
    inlier_indices: List[int]  # 内点对应的检测索引
    
    # 暗腔室信息
    dark_chamber_indices: List[int]  # 暗腔室索引列表
    chamber_brightness: Dict[int, float]  # 各腔室亮度值
    
    # 质量指标
    reprojection_error: float  # 平均重投影误差 (像素)
    fit_success: bool  # 拟合是否成功
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        return {
            'fitted_centers': self.fitted_centers.tolist(),
            'visibility': self.visibility.tolist(),
            'detected_mask': self.detected_mask.tolist(),
            'transform_matrix': self.transform_matrix.tolist(),
            'inlier_ratio': self.inlier_ratio,
            'inlier_indices': self.inlier_indices,
            'dark_chamber_indices': self.dark_chamber_indices,
            'chamber_brightness': self.chamber_brightness,
            'reprojection_error': self.reprojection_error,
            'fit_success': self.fit_success
        }


class TopologyFitter:
    """
    拓扑约束模板拟合器
    
    用法：
        config = TopologyConfig(template_scale=50.0)
        fitter = TopologyFitter(config)
        
        result = fitter.fit_and_fill(
            detected_centers=centers,
            image_shape=(height, width),
            image=original_image  # 可选，用于暗腔室判定
        )
        
        # 获取所有 12 个腔室坐标
        all_centers = result.fitted_centers
        
        # 获取暗腔室索引
        dark_indices = result.dark_chamber_indices
    """
    
    def __init__(self, config: TopologyConfig):
        """
        初始化拓扑拟合器
        
        :param config: 拓扑配置
        """
        self.config = config
        self.template = self._load_template()
        logger.info(f"TopologyFitter initialized: scale={config.template_scale}, "
                   f"template_shape={self.template.shape}")
    
    def _load_template(self) -> np.ndarray:
        """加载或生成模板点集"""
        if self.config.template_path:
            # 从 JSON 文件加载自定义模板
            try:
                template_path = Path(self.config.template_path)
                if template_path.exists():
                    with open(template_path, 'r') as f:
                        data = json.load(f)
                    template = np.array(data['template'], dtype=np.float32)
                    logger.info(f"Loaded custom template from {template_path}")
                    return template * self.config.template_scale
            except Exception as e:
                logger.warning(f"Failed to load custom template: {e}, using default")
        
        # 使用默认十字模板
        return DEFAULT_CROSS_TEMPLATE * self.config.template_scale
    
    def fit_and_fill(
        self,
        detected_centers: np.ndarray,
        image_shape: Tuple[int, int],
        image: Optional[np.ndarray] = None
    ) -> FittingResult:
        """
        RANSAC 拟合 + 回填缺失腔室
        
        :param detected_centers: (N, 2) 检测到的腔室中心点
        :param image_shape: (H, W) 图像尺寸
        :param image: 原图 (用于暗腔室亮度判定，可选)
        :return: FittingResult
        """
        h, w = image_shape
        
        # 特殊情况：检测点过少
        if len(detected_centers) < 2:
            logger.warning(f"Too few detections ({len(detected_centers)}), cannot fit")
            return self._create_failed_result(image_shape)
        
        detected_centers = np.array(detected_centers, dtype=np.float32)
        
        # Step 1: RANSAC Similarity Transform
        transform, inlier_indices, reproj_error = self._ransac_similarity(
            detected_centers, image_shape
        )
        
        if transform is None:
            logger.warning("RANSAC fit failed, using fallback")
            return self._create_failed_result(image_shape)
        
        # Step 2: 变换模板到图像坐标
        fitted_centers = self._transform_template(transform)
        
        # Step 3: 可见性判定
        visibility = self._check_visibility(fitted_centers, image_shape)
        
        # Step 4: 匹配检测点与模板点
        detected_mask = self._match_detections(detected_centers, fitted_centers)
        
        # Step 5: 暗腔室判定 (如果提供图像)
        dark_indices = []
        chamber_brightness = {}
        if image is not None:
            dark_indices, chamber_brightness = self._identify_dark_chambers(
                image, fitted_centers, visibility
            )
        
        return FittingResult(
            fitted_centers=fitted_centers,
            visibility=visibility,
            detected_mask=detected_mask,
            transform_matrix=transform,
            inlier_ratio=len(inlier_indices) / len(detected_centers) if len(detected_centers) > 0 else 0,
            inlier_indices=inlier_indices,
            dark_chamber_indices=dark_indices,
            chamber_brightness=chamber_brightness,
            reprojection_error=reproj_error,
            fit_success=True
        )
    
    def _ransac_similarity(
        self,
        src_points: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], List[int], float]:
        """
        RANSAC 拟合 Similarity Transform
        
        :return: (transform_matrix, inlier_indices, reprojection_error)
        """
        h, w = image_shape
        
        best_transform = None
        best_inliers = []
        best_error = float('inf')
        
        # 估计图像中心作为初始猜测
        img_center = np.array([w / 2, h / 2])
        
        for _ in range(self.config.ransac_iters):
            # 随机采样 2 个检测点
            if len(src_points) < 2:
                break
            
            sample_idx = np.random.choice(len(src_points), 2, replace=False)
            sample_points = src_points[sample_idx]
            
            # 为采样点找最佳匹配的模板点
            template_idx = self._find_best_template_match(sample_points, img_center)
            if template_idx is None:
                continue
            
            template_points = self.template[template_idx]
            
            # 拟合 Similarity Transform (estimateAffinePartial2D)
            try:
                M, _ = cv2.estimateAffinePartial2D(
                    template_points.reshape(-1, 1, 2),
                    sample_points.reshape(-1, 1, 2)
                )
                
                if M is None:
                    continue
                
                # 计算内点
                transformed = self._transform_template(M)
                inliers, error = self._count_inliers(src_points, transformed)
                
                if len(inliers) > len(best_inliers) or (
                    len(inliers) == len(best_inliers) and error < best_error
                ):
                    best_inliers = inliers
                    best_transform = M
                    best_error = error
                    
            except Exception as e:
                continue
        
        # 检查拟合质量
        if len(best_inliers) < self.config.min_inliers:
            logger.warning(f"Only {len(best_inliers)} inliers found, below threshold {self.config.min_inliers}")
            
            # 尝试回退到 Affine
            if self.config.fallback_to_affine and len(src_points) >= 3:
                return self._fallback_affine(src_points, image_shape)
            
            return None, [], float('inf')
        
        return best_transform, best_inliers, best_error
    
    def _find_best_template_match(
        self,
        sample_points: np.ndarray,
        img_center: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        为采样点找最佳的模板点匹配
        
        策略：基于相对位置和方向匹配
        """
        # 使用最近邻 + 相对角度匹配
        # 简化：随机选择两个相邻的模板点
        n_template = len(self.template)
        
        # 尝试多种组合
        for _ in range(10):
            idx1 = np.random.randint(n_template)
            # 选择同一个臂上的另一个点，或相邻臂
            if idx1 % 3 < 2:  # 不是臂的最外点
                idx2 = idx1 + 1
            else:
                idx2 = (idx1 - 1) % n_template
            
            return np.array([idx1, idx2])
        
        return None
    
    def _count_inliers(
        self,
        detected: np.ndarray,
        transformed_template: np.ndarray
    ) -> Tuple[List[int], float]:
        """计算内点和重投影误差"""
        inliers = []
        total_error = 0.0
        
        for i, det_pt in enumerate(detected):
            # 找最近的模板点
            dists = np.linalg.norm(transformed_template - det_pt, axis=1)
            min_dist = np.min(dists)
            
            if min_dist < self.config.ransac_threshold:
                inliers.append(i)
                total_error += min_dist
        
        avg_error = total_error / len(inliers) if inliers else float('inf')
        return inliers, avg_error
    
    def _fallback_affine(
        self,
        src_points: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], List[int], float]:
        """回退到 Affine Transform"""
        logger.info("Falling back to Affine transform")
        
        # 使用所有点的质心和分布估计变换
        centroid = np.mean(src_points, axis=0)
        template_centroid = np.array([0.0, 0.0])  # 模板中心
        
        # 估计缩放（基于点的分布范围）
        src_span = np.max(src_points, axis=0) - np.min(src_points, axis=0)
        template_span = np.max(self.template, axis=0) - np.min(self.template, axis=0)
        scale = np.mean(src_span / (template_span + 1e-6))
        
        # 简单平移 + 缩放矩阵
        M = np.array([
            [scale, 0, centroid[0]],
            [0, scale, centroid[1]]
        ], dtype=np.float32)
        
        transformed = self._transform_template(M)
        inliers, error = self._count_inliers(src_points, transformed)
        
        return M, inliers, error
    
    def _transform_template(self, M: np.ndarray) -> np.ndarray:
        """应用变换矩阵到模板"""
        # M 是 2x3 矩阵
        ones = np.ones((len(self.template), 1))
        template_homo = np.hstack([self.template, ones])  # (N, 3)
        transformed = (M @ template_homo.T).T  # (N, 2)
        return transformed
    
    def _check_visibility(
        self,
        centers: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """检查每个点是否在图像边界内"""
        h, w = image_shape
        margin = self.config.visibility_margin
        
        visibility = np.ones(len(centers), dtype=bool)
        
        for i, (cx, cy) in enumerate(centers):
            if cx < margin or cx > w - margin or cy < margin or cy > h - margin:
                visibility[i] = False
        
        return visibility
    
    def _match_detections(
        self,
        detected: np.ndarray,
        fitted: np.ndarray
    ) -> np.ndarray:
        """匹配检测点到拟合的模板点"""
        detected_mask = np.zeros(len(fitted), dtype=bool)
        
        for i, fit_pt in enumerate(fitted):
            dists = np.linalg.norm(detected - fit_pt, axis=1)
            if np.min(dists) < self.config.ransac_threshold * 1.5:
                detected_mask[i] = True
        
        return detected_mask
    
    def _identify_dark_chambers(
        self,
        image: np.ndarray,
        fitted_centers: np.ndarray,
        visibility: np.ndarray
    ) -> Tuple[List[int], Dict[int, float]]:
        """
        基于拓扑位置判定暗腔室
        
        拓扑约束：
        - 暗腔室只有 1 个
        - 暗腔室位于其所处臂的最外侧（索引 2, 5, 8, 11）
        
        策略：
        1. 在每个可见腔室位置取固定 ROI
        2. 计算 ROI 区域的平均亮度
        3. 在最外侧的 4 个腔室中，找出亮度最低的那个作为暗腔室
        """
        roi_size = self.config.brightness_roi_size
        half_size = roi_size // 2
        brightnesses = {}
        
        h, w = image.shape[:2]
        
        for i, (cx, cy) in enumerate(fitted_centers):
            if not visibility[i]:
                continue
            
            # ROI 边界
            x1 = int(max(0, cx - half_size))
            y1 = int(max(0, cy - half_size))
            x2 = int(min(w, cx + half_size))
            y2 = int(min(h, cy + half_size))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            roi = image[y1:y2, x1:x2]
            
            # 转灰度计算亮度
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            brightnesses[i] = float(np.mean(gray))
        
        # 拓扑约束：暗腔室只可能在最外侧 (索引 2, 5, 8, 11)
        if not brightnesses:
            return [], brightnesses
        
        # 在最外侧腔室中找亮度最低的
        outermost_brightness = {
            i: brightnesses[i]
            for i in OUTERMOST_CHAMBER_INDICES
            if i in brightnesses
        }
        
        if not outermost_brightness:
            logger.warning("No visible outermost chambers for dark detection")
            return [], brightnesses
        
        # 找到亮度最低的最外侧腔室
        dark_idx = min(outermost_brightness, key=outermost_brightness.get)
        
        # 如果该腔室亮度确实显著低于其他腔室，才标记为暗腔室
        brightness_values = list(brightnesses.values())
        threshold = np.percentile(brightness_values, self.config.dark_percentile)
        
        if brightnesses[dark_idx] < threshold:
            dark_indices = [dark_idx]
        else:
            dark_indices = []  # 无暗腔室（所有腔室亮度相近）
        
        logger.info(f"Dark chamber detection: outermost candidates={list(outermost_brightness.keys())}, "
                   f"selected={dark_indices}, brightness={brightnesses.get(dark_idx, 'N/A'):.1f}")
        
        return dark_indices, brightnesses
    
    def _create_failed_result(
        self,
        image_shape: Tuple[int, int]
    ) -> FittingResult:
        """创建拟合失败的结果"""
        h, w = image_shape
        
        # 放置在图像中心的默认位置
        center = np.array([w / 2, h / 2])
        default_centers = self.template + center
        
        return FittingResult(
            fitted_centers=default_centers,
            visibility=np.zeros(12, dtype=bool),
            detected_mask=np.zeros(12, dtype=bool),
            transform_matrix=np.eye(2, 3, dtype=np.float32),
            inlier_ratio=0.0,
            inlier_indices=[],
            dark_chamber_indices=[],
            chamber_brightness={},
            reprojection_error=float('inf'),
            fit_success=False
        )
    
    @staticmethod
    def save_template(template: np.ndarray, path: Path) -> None:
        """保存模板到 JSON 文件"""
        data = {
            'template': (template / 50.0).tolist(),  # 归一化保存
            'description': 'Cross template for 12 chambers, normalized coordinates'
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def __repr__(self) -> str:
        return (
            f"TopologyFitter("
            f"template_scale={self.config.template_scale}, "
            f"ransac_iters={self.config.ransac_iters})"
        )
