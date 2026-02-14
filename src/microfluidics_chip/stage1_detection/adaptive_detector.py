"""
自适应粗到精检测器

实现 "global_scan -> cluster_roi -> fine_scan" 流程，
解决远近尺度变化和暗腔室漏检问题。

核心流程：
1. Global Scan: 低分辨率粗扫描，低置信度阈值，捕获候选点
2. Cluster ROI: DBSCAN 聚类找最密集区域，生成自适应 ROI  
3. Preprocess ROI: 可选 CLAHE 预处理
4. Fine Scan: 高分辨率精细扫描
5. Coordinate Mapping: ROI 坐标映射回原图

设计原则：
- 不修改原有 ChamberDetector，通过组合方式扩展
- 所有配置可注入，便于实验调参
- 支持回退机制：聚类失败时使用全图
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN

from ..core.types import ChamberDetection
from ..core.logger import get_logger
from .detector import ChamberDetector
from .preprocess import preprocess_image, extract_roi

logger = get_logger("stage1_detection.adaptive_detector")


@dataclass
class ClusterResult:
    """聚类结果"""
    roi_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 原图坐标
    cluster_centers: List[Tuple[float, float]]  # 聚类内点的中心
    cluster_score: float  # 聚类质量评分 (0-1)
    is_fallback: bool = False  # 是否使用了回退策略
    num_clusters_found: int = 0  # 发现的簇数量


@dataclass
class AdaptiveDetectionConfig:
    """自适应检测配置"""
    # 粗扫描参数
    coarse_imgsz: int = 640
    coarse_conf: float = 0.08
    
    # 精细扫描参数
    fine_imgsz: int = 1280
    fine_conf: float = 0.3
    
    # 聚类参数
    cluster_eps: float = 100.0  # DBSCAN eps (像素距离)
    cluster_min_samples: int = 3  # DBSCAN min_samples
    
    # ROI 参数
    roi_margin: float = 1.3  # ROI 扩展系数 (1.0 = 无扩展)
    min_roi_size: int = 200  # 最小 ROI 尺寸
    
    # 预处理参数
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0


class AdaptiveDetector:
    """
    自适应粗到精检测器
    
    用法：
        config = AdaptiveDetectionConfig()
        detector = ChamberDetector(yolo_config)
        adaptive = AdaptiveDetector(config, detector)
        
        detections, cluster_result = adaptive.detect_adaptive(image)
    """
    
    def __init__(
        self,
        config: AdaptiveDetectionConfig,
        detector: ChamberDetector
    ):
        """
        初始化自适应检测器
        
        :param config: 自适应检测配置
        :param detector: 基础 YOLO 检测器实例
        """
        self.config = config
        self.detector = detector
        logger.info(f"AdaptiveDetector initialized: coarse_conf={config.coarse_conf}, fine_conf={config.fine_conf}")
    
    def detect_adaptive(
        self,
        image: np.ndarray
    ) -> Tuple[List[ChamberDetection], ClusterResult]:
        """
        自适应检测主入口
        
        :param image: 原始图像 (H, W, 3) BGR uint8
        :return: (detections, cluster_result)
                 - detections: 精细检测结果列表 (原图坐标系)
                 - cluster_result: 聚类信息
        """
        h, w = image.shape[:2]
        logger.info(f"Starting adaptive detection on image {w}x{h}")
        
        # Step 1: Global coarse scan
        coarse_dets = self._global_scan(image)
        logger.info(f"Coarse scan: {len(coarse_dets)} detections")
        
        # Step 2: Cluster ROI
        cluster_result = self._cluster_roi(coarse_dets, (h, w))
        logger.info(f"Cluster ROI: bbox={cluster_result.roi_bbox}, score={cluster_result.cluster_score:.2f}")
        
        # Step 3: Extract and preprocess ROI
        roi_image = self._extract_and_preprocess_roi(image, cluster_result.roi_bbox)
        
        # Step 4: Fine scan on ROI
        fine_dets = self._fine_scan(roi_image)
        logger.info(f"Fine scan: {len(fine_dets)} detections")
        
        # Step 5: Map back to original coordinates
        mapped_dets = self._map_to_original(fine_dets, cluster_result.roi_bbox, roi_image.shape[:2])
        logger.info(f"Mapped to original: {len(mapped_dets)} detections")
        
        return mapped_dets, cluster_result
    
    def _global_scan(self, image: np.ndarray) -> List[ChamberDetection]:
        """
        低分辨率粗扫描
        
        使用较低置信度阈值捕获更多候选点
        """
        h, w = image.shape[:2]
        
        # 调整到粗扫描分辨率
        scale = self.config.coarse_imgsz / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # 临时修改置信度阈值
        original_conf = self.detector.config.confidence_threshold
        self.detector.config.confidence_threshold = self.config.coarse_conf
        
        try:
            dets = self.detector.detect(resized)
        finally:
            # 恢复原始阈值
            self.detector.config.confidence_threshold = original_conf
        
        # 映射回原图坐标
        return self._scale_detections(dets, 1.0/scale, 1.0/scale)
    
    def _cluster_roi(
        self,
        detections: List[ChamberDetection],
        image_shape: Tuple[int, int]
    ) -> ClusterResult:
        """
        DBSCAN 聚类 + 外接矩形扩展
        
        策略：
        1. 如果检测点过少，回退到全图
        2. 使用 DBSCAN 找最密集的点簇
        3. 取该簇的外接矩形并按 margin 扩展
        """
        h, w = image_shape
        
        # 回退条件：检测点过少
        if len(detections) < self.config.cluster_min_samples:
            logger.warning(f"Too few detections ({len(detections)}), using full image as ROI")
            return ClusterResult(
                roi_bbox=(0, 0, w, h),
                cluster_centers=[d.center for d in detections],
                cluster_score=0.0,
                is_fallback=True,
                num_clusters_found=0
            )
        
        # 提取中心点
        centers = np.array([d.center for d in detections])
        
        # DBSCAN 聚类
        clustering = DBSCAN(
            eps=self.config.cluster_eps,
            min_samples=self.config.cluster_min_samples
        ).fit(centers)
        
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}  # 排除噪声点
        
        # 无有效簇：回退到所有点的 bbox
        if not unique_labels:
            logger.warning("No valid clusters found, using all detections bbox")
            return self._fallback_cluster(detections, image_shape)
        
        # 找最大簇
        best_label = max(unique_labels, key=lambda l: np.sum(labels == l))
        cluster_mask = labels == best_label
        cluster_centers = centers[cluster_mask]
        
        # 计算外接矩形
        x_min, y_min = cluster_centers.min(axis=0)
        x_max, y_max = cluster_centers.max(axis=0)
        
        # 应用 margin 扩展
        box_w = x_max - x_min
        box_h = y_max - y_min
        margin = self.config.roi_margin
        
        # 计算扩展量
        expand_w = box_w * (margin - 1) / 2
        expand_h = box_h * (margin - 1) / 2
        
        # 确保最小尺寸
        expand_w = max(expand_w, self.config.min_roi_size / 2)
        expand_h = max(expand_h, self.config.min_roi_size / 2)
        
        x1 = int(max(0, x_min - expand_w))
        y1 = int(max(0, y_min - expand_h))
        x2 = int(min(w, x_max + expand_w))
        y2 = int(min(h, y_max + expand_h))
        
        return ClusterResult(
            roi_bbox=(x1, y1, x2, y2),
            cluster_centers=[tuple(c) for c in cluster_centers.tolist()],
            cluster_score=len(cluster_centers) / len(detections),
            is_fallback=False,
            num_clusters_found=len(unique_labels)
        )
    
    def _fallback_cluster(
        self,
        detections: List[ChamberDetection],
        image_shape: Tuple[int, int]
    ) -> ClusterResult:
        """回退策略：使用所有检测点的外接矩形"""
        h, w = image_shape
        
        if not detections:
            return ClusterResult(
                roi_bbox=(0, 0, w, h),
                cluster_centers=[],
                cluster_score=0.0,
                is_fallback=True,
                num_clusters_found=0
            )
        
        centers = np.array([d.center for d in detections])
        x_min, y_min = centers.min(axis=0)
        x_max, y_max = centers.max(axis=0)
        
        # 应用 margin
        box_w = max(x_max - x_min, self.config.min_roi_size)
        box_h = max(y_max - y_min, self.config.min_roi_size)
        margin = self.config.roi_margin
        
        x1 = int(max(0, x_min - box_w * (margin - 1) / 2))
        y1 = int(max(0, y_min - box_h * (margin - 1) / 2))
        x2 = int(min(w, x_max + box_w * (margin - 1) / 2))
        y2 = int(min(h, y_max + box_h * (margin - 1) / 2))
        
        return ClusterResult(
            roi_bbox=(x1, y1, x2, y2),
            cluster_centers=[tuple(c) for c in centers.tolist()],
            cluster_score=1.0,  # 全部点都在
            is_fallback=True,
            num_clusters_found=0
        )
    
    def _extract_and_preprocess_roi(
        self,
        image: np.ndarray,
        roi_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """提取 ROI 并应用预处理"""
        x1, y1, x2, y2 = roi_bbox
        roi = image[y1:y2, x1:x2].copy()
        
        # 应用预处理
        if self.config.enable_clahe:
            roi = preprocess_image(
                roi,
                enable_clahe=True,
                enable_invert=False,
                clahe_clip_limit=self.config.clahe_clip_limit
            )
        
        return roi
    
    def _fine_scan(self, roi_image: np.ndarray) -> List[ChamberDetection]:
        """
        高分辨率精细扫描
        
        在 ROI 图像上使用更高置信度阈值进行检测
        """
        h, w = roi_image.shape[:2]
        
        # 如果 ROI 太小，可能需要放大
        target_size = self.config.fine_imgsz
        if max(h, w) < target_size:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            roi_resized = cv2.resize(roi_image, (new_w, new_h))
            needs_scale_back = True
        else:
            roi_resized = roi_image
            scale = 1.0
            needs_scale_back = False
        
        # 使用精细扫描置信度
        original_conf = self.detector.config.confidence_threshold
        self.detector.config.confidence_threshold = self.config.fine_conf
        
        try:
            dets = self.detector.detect(roi_resized)
        finally:
            self.detector.config.confidence_threshold = original_conf
        
        # 如果放大过，需要缩放回来
        if needs_scale_back:
            dets = self._scale_detections(dets, 1.0/scale, 1.0/scale)
        
        return dets
    
    def _map_to_original(
        self,
        detections: List[ChamberDetection],
        roi_bbox: Tuple[int, int, int, int],
        roi_shape: Tuple[int, int]
    ) -> List[ChamberDetection]:
        """
        将 ROI 坐标系的检测结果映射回原图坐标系
        """
        x1, y1, x2, y2 = roi_bbox
        
        mapped = []
        for det in detections:
            # 偏移中心点
            new_cx = det.center[0] + x1
            new_cy = det.center[1] + y1
            
            # 偏移 bbox
            old_x, old_y, old_w, old_h = det.bbox
            new_bbox = (old_x + x1, old_y + y1, old_w, old_h)
            
            mapped.append(ChamberDetection(
                bbox=new_bbox,
                center=(new_cx, new_cy),
                class_id=det.class_id,
                confidence=det.confidence
            ))
        
        return mapped
    
    def _scale_detections(
        self,
        detections: List[ChamberDetection],
        scale_x: float,
        scale_y: float
    ) -> List[ChamberDetection]:
        """缩放检测结果坐标"""
        scaled = []
        for det in detections:
            new_cx = det.center[0] * scale_x
            new_cy = det.center[1] * scale_y
            
            old_x, old_y, old_w, old_h = det.bbox
            new_bbox = (
                int(old_x * scale_x),
                int(old_y * scale_y),
                int(old_w * scale_x),
                int(old_h * scale_y)
            )
            
            scaled.append(ChamberDetection(
                bbox=new_bbox,
                center=(new_cx, new_cy),
                class_id=det.class_id,
                confidence=det.confidence
            ))
        
        return scaled
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveDetector("
            f"coarse_conf={self.config.coarse_conf}, "
            f"fine_conf={self.config.fine_conf}, "
            f"cluster_eps={self.config.cluster_eps})"
        )
