"""
测试 Output 类型可 JSON 序列化（P1验证）
验证路径字段为 str 类型（不包含 numpy）
"""

import pytest
import json
import numpy as np
from microfluidics_chip.core.types import (
    Stage1Output,
    Stage2Output,
    TransformParams,
    ChamberDetection
)


def test_stage1_output_paths_are_str():
    """P1验证：验证 Stage1Output 路径字段为 str 类型"""
    output = Stage1Output(
        chip_id="test_chip",
        aligned_image_path="aligned.png",
        chamber_slices_path="chamber_slices.npz",
        transform_params=TransformParams(
            rotation_angle=45.0,
            scale_factor=1.2,
            chip_centroid=(300.0, 300.0),
            blank_arm_index=0
        ),
        num_chambers=12,
        processing_time=1.5
    )
    
    # 验证类型
    assert isinstance(output.aligned_image_path, str)
    assert isinstance(output.chamber_slices_path, str)
    assert output.aligned_image_path == "aligned.png"
    assert output.chamber_slices_path == "chamber_slices.npz"


def test_stage1_output_serialization():
    """验证 Stage1Output 可 JSON 序列化"""
    output = Stage1Output(
        chip_id="test_chip",
        aligned_image_path="aligned.png",
        chamber_slices_path="chamber_slices.npz",
        transform_params=TransformParams(
            rotation_angle=45.0,
            scale_factor=1.2,
            chip_centroid=(300.0, 300.0),
            blank_arm_index=0
        ),
        num_chambers=12,
        processing_time=1.5
    )
    
    # 序列化
    json_str = output.model_dump_json()
    data = json.loads(json_str)
    
    # 验证关键字段
    assert data['chip_id'] == "test_chip"
    assert data['aligned_image_path'] == "aligned.png"  # 相对路径
    assert data['chamber_slices_path'] == "chamber_slices.npz"
    
    # 反序列化
    output_loaded = Stage1Output(**data)
    assert output_loaded.chip_id == "test_chip"
    assert output_loaded.num_chambers == 12


def test_stage2_output_serialization():
    """验证 Stage2Output 可 JSON 序列化"""
    output = Stage2Output(
        chip_id="test_chip",
        corrected_slices_path="corrected_slices.npz",
        processing_time=2.0
    )
    
    json_str = output.model_dump_json()
    data = json.loads(json_str)
    
    assert data['corrected_slices_path'] == "corrected_slices.npz"


def test_output_no_numpy():
    """验证 Output 不包含 numpy 数组"""
    output = Stage1Output(
        chip_id="test",
        transform_params=TransformParams(
            rotation_angle=0.0,
            scale_factor=1.0,
            chip_centroid=(0.0, 0.0),
            blank_arm_index=0
        ),
        num_chambers=12
    )
    
    # 转为 dict
    data = output.model_dump()
    
    # 递归检查无 numpy
    def check_no_numpy(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                check_no_numpy(v)
        elif isinstance(obj, list):
            for item in obj:
                check_no_numpy(item)
        else:
            assert not isinstance(obj, np.ndarray), "Output 包含 numpy 数组！"
    
    check_no_numpy(data)


def test_chamber_detection_type():
    """P0验证：验证 ChamberDetection 强类型"""
    detection = ChamberDetection(
        bbox=(100, 100, 50, 50),
        center=(125.0, 125.0),
        class_id=1,
        confidence=0.95
    )
    
    assert detection.bbox == (100, 100, 50, 50)
    assert detection.center == (125.0, 125.0)
    assert detection.class_id == 1
    assert detection.confidence == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
