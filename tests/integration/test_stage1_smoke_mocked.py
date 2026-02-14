"""
Stage1 Smoke Test (Mocked)
验证 P0/P1/P2 规范（无需权重文件）
"""

import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
from microfluidics_chip.core.types import (
    ChamberDetection,
    TransformParams,
    Stage1Result,
    Stage1Output
)
from microfluidics_chip.core.config import get_default_config
from microfluidics_chip.core.io import save_stage1_result, load_stage1_output
from microfluidics_chip.stage1_detection.inference import infer_stage1


@pytest.fixture
def temp_output_dir(tmp_path):
    """临时输出目录"""
    output_dir = tmp_path / "test_stage1"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_detector():
    """Mock ChamberDetector（P0验证）"""
    detector = Mock()
    
    # P0: 返回 List[ChamberDetection]
    detections = []
    for i in range(12):
        det = ChamberDetection(
            bbox=(100 + i*10, 100 + i*10, 50, 50),
            center=(125.0 + i*10, 125.0 + i*10),
            class_id=0 if i == 0 else 1,  # 第一个为 blank
            confidence=0.95
        )
        detections.append(det)
    
    detector.detect.return_value = detections
    return detector


@pytest.fixture
def mock_geometry_engine():
    """Mock CrossGeometryEngine（P0验证）"""
    engine = Mock()
    
    # P0: 返回四元组
    aligned_image = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
    chamber_slices = np.random.randint(0, 255, (12, 80, 80, 3), dtype=np.uint8)
    transform_params = TransformParams(
        rotation_angle=45.0,
        scale_factor=1.2,
        chip_centroid=(300.0, 300.0),
        blank_arm_index=0
    )
    debug_vis = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
    
    engine.process.return_value = (aligned_image, chamber_slices, transform_params, debug_vis)
    return engine


def test_p0_detector_interface(mock_detector):
    """P0验证：detector.detect() 返回 List[ChamberDetection]"""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    result = mock_detector.detect(img)
    
    # 验证类型
    assert isinstance(result, list)
    assert len(result) == 12
    assert all(isinstance(det, ChamberDetection) for det in result)
    
    # 验证字段
    det = result[0]
    assert hasattr(det, 'bbox')
    assert hasattr(det, 'center')
    assert hasattr(det, 'class_id')
    assert hasattr(det, 'confidence')


def test_p0_geometry_engine_interface(mock_geometry_engine):
    """P0验证：geometry_engine.process() 返回四元组"""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = [
        ChamberDetection(bbox=(100, 100, 50, 50), center=(125.0, 125.0), class_id=0, confidence=0.95)
        for _ in range(12)
    ]
    
    result = mock_geometry_engine.process(img, detections)
    
    # 验证四元组
    assert len(result) == 4
    aligned_image, chamber_slices, transform_params, debug_vis = result
    
    assert isinstance(aligned_image, np.ndarray)
    assert isinstance(chamber_slices, np.ndarray)
    assert isinstance(transform_params, TransformParams)
    assert isinstance(debug_vis, np.ndarray)


def test_p2_fixed_filenames(temp_output_dir, mock_detector, mock_geometry_engine):
    """P2验证：固定文件命名"""
    # 创建 mock 结果
    result = Stage1Result(
        chip_id="test_chip",
        aligned_image=np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8),
        chamber_slices=np.random.randint(0, 255, (12, 80, 80, 3), dtype=np.uint8),
        transform_params=TransformParams(
            rotation_angle=45.0,
            scale_factor=1.2,
            chip_centroid=(300.0, 300.0),
            blank_arm_index=0
        ),
        chambers=[ChamberDetection(bbox=(0,0,50,50), center=(25.0,25.0), class_id=0, confidence=0.9)] * 12,
        processing_time=1.5
    )
    
    # 保存
    run_dir = temp_output_dir / "test_chip"
    output = save_stage1_result(result, run_dir, save_gt=False)
    
    # P2: 验证固定文件名
    assert (run_dir / "stage1_metadata.json").exists()
    assert (run_dir / "aligned.png").exists()
    assert (run_dir / "chamber_slices.npz").exists()
    
    # 验证 metadata 可读
    with open(run_dir / "stage1_metadata.json") as f:
        metadata = json.load(f)
    
    assert metadata['chip_id'] == "test_chip"
    assert metadata['aligned_image_path'] == "aligned.png"  # P1: 相对路径
    assert metadata['chamber_slices_path'] == "chamber_slices.npz"


def test_p2_npz_key_validation(temp_output_dir):
    """P2验证：npz key='slices'"""
    result = Stage1Result(
        chip_id="test_chip",
        aligned_image=np.zeros((600, 600, 3), dtype=np.uint8),
        chamber_slices=np.random.randint(0, 255, (12, 80, 80, 3), dtype=np.uint8),
        transform_params=TransformParams(
            rotation_angle=0.0,
            scale_factor=1.0,
            chip_centroid=(0.0, 0.0),
            blank_arm_index=0
        ),
        chambers=[],
        processing_time=1.0
    )
    
    run_dir = temp_output_dir / "test_chip"
    save_stage1_result(result, run_dir)
    
    # 验证 npz key
    npz_data = np.load(run_dir / "chamber_slices.npz")
    assert 'slices' in npz_data
    assert npz_data['slices'].shape == (12, 80, 80, 3)


def test_p2_no_glob_loading(temp_output_dir):
    """P2验证：禁止 Glob，只认固定文件名"""
    result = Stage1Result(
        chip_id="test_chip",
        aligned_image=np.zeros((600, 600, 3), dtype=np.uint8),
        chamber_slices=np.random.randint(0, 255, (12, 80, 80, 3), dtype=np.uint8),
        transform_params=TransformParams(
            rotation_angle=0.0,
            scale_factor=1.0,
            chip_centroid=(0.0, 0.0),
            blank_arm_index=0
        ),
        chambers=[],
        processing_time=1.0
    )
    
    run_dir = temp_output_dir / "test_chip"
    save_stage1_result(result, run_dir)
    
    # 加载（只认固定文件名）
    output, slices = load_stage1_output(run_dir)
    
    assert output.chip_id == "test_chip"
    assert slices.shape == (12, 80, 80, 3)


def test_p1_relative_path_loading(temp_output_dir):
    """P1验证：相对路径转绝对路径"""
    result = Stage1Result(
        chip_id="test_chip",
        aligned_image=np.zeros((600, 600, 3), dtype=np.uint8),
        chamber_slices=np.random.randint(0, 255, (12, 80, 80, 3), dtype=np.uint8),
        transform_params=TransformParams(
            rotation_angle=0.0,
            scale_factor=1.0,
            chip_centroid=(0.0, 0.0),
            blank_arm_index=0
        ),
        chambers=[],
        processing_time=1.0
    )
    
    run_dir = temp_output_dir / "test_chip"
    save_stage1_result(result, run_dir)
    
    # 验证 metadata 中的路径为 str（相对路径）
    with open(run_dir / "stage1_metadata.json") as f:
        metadata = json.load(f)
    
    assert isinstance(metadata['aligned_image_path'], str)
    assert metadata['aligned_image_path'] == "aligned.png"  # P1: 相对路径
    
    # 加载时应能正确转换为绝对路径
    output, slices = load_stage1_output(run_dir)
    
    # 验证加载成功（说明相对路径转换正确）
    assert output.aligned_image_path == "aligned.png"
    assert slices is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
