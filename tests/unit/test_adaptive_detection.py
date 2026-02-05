"""
自适应检测模块单元测试

测试覆盖：
- 聚类 ROI 计算
- 坐标映射
- RANSAC 拟合
- 暗腔室判定
- 预处理函数
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# 测试导入
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPreprocess:
    """预处理函数测试"""
    
    def test_apply_clahe(self):
        """测试 CLAHE 应用"""
        from microfluidics_chip.stage1_detection.preprocess import apply_clahe
        
        # 创建测试图像
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = apply_clahe(image)
        
        assert result.shape == image.shape
        assert result.dtype == np.uint8
    
    def test_apply_clahe_empty_image(self):
        """测试空图像"""
        from microfluidics_chip.stage1_detection.preprocess import apply_clahe
        
        image = np.array([])
        result = apply_clahe(image)
        
        assert result.size == 0
    
    def test_apply_invert(self):
        """测试亮度反转"""
        from microfluidics_chip.stage1_detection.preprocess import apply_invert
        
        image = np.array([[[0, 128, 255]]], dtype=np.uint8)
        result = apply_invert(image)
        
        expected = np.array([[[255, 127, 0]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
    
    def test_preprocess_pipeline(self):
        """测试预处理流水线"""
        from microfluidics_chip.stage1_detection.preprocess import preprocess_image
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # 仅 CLAHE
        result1 = preprocess_image(image, enable_clahe=True, enable_invert=False)
        assert result1.shape == image.shape
        
        # 仅 Invert
        result2 = preprocess_image(image, enable_clahe=False, enable_invert=True)
        assert result2.shape == image.shape
        
        # 两者都开
        result3 = preprocess_image(image, enable_clahe=True, enable_invert=True)
        assert result3.shape == image.shape


class TestAdaptiveDetector:
    """自适应检测器测试"""
    
    def test_cluster_result_dataclass(self):
        """测试 ClusterResult 数据类"""
        from microfluidics_chip.stage1_detection.adaptive_detector import ClusterResult
        
        result = ClusterResult(
            roi_bbox=(0, 0, 100, 100),
            cluster_centers=[(50, 50)],
            cluster_score=0.9,
            is_fallback=False,
            num_clusters_found=1
        )
        
        assert result.roi_bbox == (0, 0, 100, 100)
        assert result.cluster_score == 0.9
        assert not result.is_fallback
    
    def test_scale_detections(self):
        """测试检测结果缩放"""
        from microfluidics_chip.stage1_detection.adaptive_detector import AdaptiveDetector, AdaptiveDetectionConfig
        from microfluidics_chip.core.types import ChamberDetection
        
        # 创建 mock detector
        mock_detector = Mock()
        config = AdaptiveDetectionConfig()
        
        adaptive = AdaptiveDetector(config, mock_detector)
        
        # 创建测试检测
        detections = [
            ChamberDetection(
                bbox=(10, 20, 30, 40),
                center=(25.0, 40.0),
                class_id=0,
                confidence=0.9
            )
        ]
        
        # 缩放 2x
        scaled = adaptive._scale_detections(detections, 2.0, 2.0)
        
        assert len(scaled) == 1
        assert scaled[0].center == (50.0, 80.0)
        assert scaled[0].bbox == (20, 40, 60, 80)


class TestTopologyFitter:
    """拓扑拟合器测试"""
    
    def test_default_template(self):
        """测试默认模板加载"""
        from microfluidics_chip.stage1_detection.topology_fitter import TopologyFitter, TopologyConfig
        
        config = TopologyConfig(template_scale=50.0)
        fitter = TopologyFitter(config)
        
        assert fitter.template.shape == (12, 2)
    
    def test_visibility_check(self):
        """测试可见性判定"""
        from microfluidics_chip.stage1_detection.topology_fitter import TopologyFitter, TopologyConfig
        
        config = TopologyConfig(visibility_margin=10)
        fitter = TopologyFitter(config)
        
        # 测试点
        centers = np.array([
            [50, 50],    # 可见
            [5, 50],     # 太靠左
            [50, 5],     # 太靠上
            [95, 50],    # 太靠右
            [50, 95],    # 太靠下
        ])
        
        visibility = fitter._check_visibility(centers, (100, 100))
        
        assert visibility[0] == True   # 中心点可见
        assert visibility[1] == False  # 边界外
        assert visibility[2] == False
        assert visibility[3] == False
        assert visibility[4] == False
    
    def test_fit_and_fill_few_points(self):
        """测试检测点过少的情况"""
        from microfluidics_chip.stage1_detection.topology_fitter import TopologyFitter, TopologyConfig
        
        config = TopologyConfig()
        fitter = TopologyFitter(config)
        
        # 只有 1 个点
        detected = np.array([[100, 100]])
        result = fitter.fit_and_fill(detected, (500, 500))
        
        assert result.fit_success == False
        assert result.fitted_centers.shape == (12, 2)
    
    def test_transform_template(self):
        """测试模板变换"""
        from microfluidics_chip.stage1_detection.topology_fitter import TopologyFitter, TopologyConfig
        
        config = TopologyConfig(template_scale=10.0)
        fitter = TopologyFitter(config)
        
        # 简单平移矩阵
        M = np.array([
            [1, 0, 100],
            [0, 1, 200]
        ], dtype=np.float32)
        
        transformed = fitter._transform_template(M)
        
        # 检查变换后的第一个点
        expected_first = fitter.template[0] + np.array([100, 200])
        np.testing.assert_array_almost_equal(transformed[0], expected_first)


class TestConfigTypes:
    """配置类型测试"""
    
    def test_adaptive_detection_config_defaults(self):
        """测试自适应检测配置默认值"""
        from microfluidics_chip.core.config import AdaptiveDetectionConfig
        
        config = AdaptiveDetectionConfig()
        
        assert config.coarse_imgsz == 640
        assert config.coarse_conf == 0.08
        assert config.fine_imgsz == 1280
        assert config.enable_clahe == True
    
    def test_topology_config_defaults(self):
        """测试拓扑配置默认值"""
        from microfluidics_chip.core.config import TopologyConfig
        
        config = TopologyConfig()
        
        assert config.template_scale == 50.0
        assert config.ransac_iters == 200
        assert config.dark_percentile == 25.0


class TestModuleExports:
    """模块导出测试"""
    
    def test_stage1_detection_exports(self):
        """测试 stage1_detection 模块导出"""
        from microfluidics_chip.stage1_detection import (
            ChamberDetector,
            CrossGeometryEngine,
            infer_stage1,
            AdaptiveDetector,
            TopologyFitter,
            apply_clahe,
            preprocess_image
        )
        
        # 验证类型存在
        assert ChamberDetector is not None
        assert AdaptiveDetector is not None
        assert TopologyFitter is not None
    
    def test_types_exports(self):
        """测试 types 模块导出"""
        from microfluidics_chip.core.types import (
            ChamberDetection,
            AdaptiveDetectionResult
        )
        
        assert ChamberDetection is not None
        assert AdaptiveDetectionResult is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
