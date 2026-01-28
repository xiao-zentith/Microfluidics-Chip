"""
Stage2 Smoke Test (Mocked)
验证 P2 规范（无需权重文件）
"""

import pytest
import numpy as np
import json
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from microfluidics_chip.core.types import Stage2Result, Stage2Output
from microfluidics_chip.core.config import get_default_config
from microfluidics_chip.core.io import save_stage2_result, load_stage2_output
from microfluidics_chip.stage2_correction.models.dual_stream_unet import RefGuidedUNet
from microfluidics_chip.stage2_correction.losses import ROIWeightedLoss


@pytest.fixture
def temp_output_dir(tmp_path):
    """临时输出目录"""
    output_dir = tmp_path / "test_stage2"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_chamber_slices():
    """Mock 切片数据"""
    # (12, 80, 80, 3) uint8
    slices = np.random.randint(0, 255, (12, 80, 80, 3), dtype=np.uint8)
    return slices


def test_p2_fixed_filenames(temp_output_dir, mock_chamber_slices):
    """P2验证：固定文件命名"""
    # 创建 mock 结果
    result = Stage2Result(
        chip_id="test_chip",
        corrected_slices=mock_chamber_slices,
        processing_time=1.5
    )
    
    # 保存
    run_dir = temp_output_dir / "test_chip"
    output = save_stage2_result(result, run_dir)
    
    # P2: 验证固定文件名
    assert (run_dir / "stage2_metadata.json").exists()
    assert (run_dir / "corrected_slices.npz").exists()
    
    # 验证 metadata 可读
    with open(run_dir / "stage2_metadata.json") as f:
        metadata = json.load(f)
    
    assert metadata['chip_id'] == "test_chip"
    assert metadata['corrected_slices_path'] == "corrected_slices.npz"  # P1: 相对路径


def test_p2_npz_key_validation(temp_output_dir, mock_chamber_slices):
    """P2验证：npz key='slices'"""
    result = Stage2Result(
        chip_id="test_chip",
        corrected_slices=mock_chamber_slices,
        processing_time=1.0
    )
    
    run_dir = temp_output_dir / "test_chip"
    save_stage2_result(result, run_dir)
    
    # 验证 npz key
    npz_data = np.load(run_dir / "corrected_slices.npz")
    assert 'slices' in npz_data
    assert npz_data['slices'].shape == (12, 80, 80, 3)


def test_p2_no_glob_loading(temp_output_dir, mock_chamber_slices):
    """P2验证：禁止 Glob，只认固定文件名"""
    result = Stage2Result(
        chip_id="test_chip",
        corrected_slices=mock_chamber_slices,
        processing_time=1.0
    )
    
    run_dir = temp_output_dir / "test_chip"
    save_stage2_result(result, run_dir)
    
    # 加载（只认固定文件名）
    output, slices = load_stage2_output(run_dir)
    
    assert output.chip_id == "test_chip"
    assert slices.shape == (12, 80, 80, 3)


def test_p1_relative_path_loading(temp_output_dir, mock_chamber_slices):
    """P1验证：相对路径转绝对路径"""
    result = Stage2Result(
        chip_id="test_chip",
        corrected_slices=mock_chamber_slices,
        processing_time=1.0
    )
    
    run_dir = temp_output_dir / "test_chip"
    save_stage2_result(result, run_dir)
    
    # 验证 metadata 中的路径为 str（相对路径）
    with open(run_dir / "stage2_metadata.json") as f:
        metadata = json.load(f)
    
    assert isinstance(metadata['corrected_slices_path'], str)
    assert metadata['corrected_slices_path'] == "corrected_slices.npz"  # P1: 相对路径
    
    # 加载时应能正确转换为绝对路径
    output, slices = load_stage2_output(run_dir)
    
    # 验证加载成功（说明相对路径转换正确）
    assert output.corrected_slices_path == "corrected_slices.npz"
    assert slices is not None


def test_unet_model_structure():
    """测试 UNet 模型结构"""
    model = RefGuidedUNet(in_channels=3, out_channels=3, features=[64, 128, 256, 512])
    
    # 测试前向传播
    signal = torch.randn(2, 3, 80, 80)
    ref = torch.randn(2, 3, 80, 80)
    
    output = model(signal, ref)
    
    # 验证输出形状
    assert output.shape == (2, 3, 80, 80)
    
    # 验证输出范围（sigmoid 后应在 [0, 1]）
    assert output.min() >= 0
    assert output.max() <= 1


def test_roi_weighted_loss():
    """测试 ROI 加权损失"""
    loss_fn = ROIWeightedLoss(roi_radius=20, edge_weight=0.1, lambda_cos=0.2)
    
    pred = torch.rand(4, 3, 80, 80)
    target = torch.rand(4, 3, 80, 80)
    
    total_loss, pixel_loss, cos_loss = loss_fn(pred, target)
    
    # 验证损失值
    assert total_loss.item() >= 0
    assert pixel_loss.item() >= 0
    assert cos_loss.item() >= 0
    
    # 验证损失组合
    expected_total = pixel_loss + 0.2 * cos_loss
    assert torch.allclose(total_loss, expected_total, atol=1e-6)


def test_dynamic_weight_map():
    """测试动态权重图生成"""
    loss_fn = ROIWeightedLoss(roi_radius=20, edge_weight=0.1)
    
    # 测试不同尺寸
    pred1 = torch.rand(2, 3, 80, 80)
    target1 = torch.rand(2, 3, 80, 80)
    
    pred2 = torch.rand(2, 3, 64, 64)
    target2 = torch.rand(2, 3, 64, 64)
    
    # 应该能处理不同尺寸（动态生成权重图）
    loss1, _, _ = loss_fn(pred1, target1)
    loss2, _, _ = loss_fn(pred2, target2)
    
    assert loss1.item() >= 0
    assert loss2.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
