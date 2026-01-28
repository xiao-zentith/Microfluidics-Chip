"""
ResultSaver 单元测试
验证 P1/P2 规范（固定文件名、相对路径）
"""

import pytest
import numpy as np
import json
from pathlib import Path
from microfluidics_chip.core.io import ResultSaver


@pytest.fixture
def temp_dir(tmp_path):
    """临时目录"""
    return tmp_path / "test_saver"


def test_save_npz(temp_dir):
    """测试 npz 保存"""
    saver = ResultSaver(temp_dir)
    
    data = {"slices": np.random.rand(12, 80, 80, 3)}
    path = saver.save_npz("test.npz", data)
    
    assert path.exists()
    assert path.name == "test.npz"
    
    # 验证可加载
    loaded = np.load(path)
    assert 'slices' in loaded
    np.testing.assert_array_equal(loaded['slices'], data['slices'])


def test_load_npz(temp_dir):
    """测试 npz 加载"""
    saver = ResultSaver(temp_dir)
    
    data = {"slices": np.random.rand(12, 80, 80, 3)}
    saver.save_npz("test.npz", data)
    
    # 加载
    loaded_slices = saver.load_npz("test.npz", key="slices")
    
    np.testing.assert_array_equal(loaded_slices, data['slices'])


def test_save_image(temp_dir):
    """测试图像保存"""
    saver = ResultSaver(temp_dir)
    
    img = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
    path = saver.save_image("test.png", img)
    
    assert path.exists()
    assert path.name == "test.png"


def test_save_json(temp_dir):
    """测试 JSON 保存"""
    saver = ResultSaver(temp_dir)
    
    data = {
        "chip_id": "test",
        "path": "test.png",
        "count": 12
    }
    
    path = saver.save_json("metadata.json", data)
    
    assert path.exists()
    assert path.name == "metadata.json"
    
    # 验证可加载
    with open(path) as f:
        loaded = json.load(f)
    
    assert loaded == data


def test_load_json(temp_dir):
    """测试 JSON 加载"""
    saver = ResultSaver(temp_dir)
    
    data = {"key": "value", "number": 42}
    saver.save_json("test.json", data)
    
    # 加载
    loaded = saver.load_json("test.json")
    
    assert loaded == data


def test_p2_fixed_filenames(temp_dir):
    """P2验证：固定文件名（无前缀/后缀/时间戳）"""
    saver = ResultSaver(temp_dir)
    
    # 保存多次，文件名应保持一致（覆盖）
    data1 = {"slices": np.ones((12, 80, 80, 3))}
    data2 = {"slices": np.zeros((12, 80, 80, 3))}
    
    path1 = saver.save_npz("fixed_name.npz", data1)
    path2 = saver.save_npz("fixed_name.npz", data2)
    
    # 文件名应完全一致
    assert path1 == path2
    
    # 最后保存的数据应覆盖之前的
    loaded = saver.load_npz("fixed_name.npz", key="slices")
    np.testing.assert_array_equal(loaded, data2['slices'])


def test_p1_relative_path_storage(temp_dir):
    """P1验证：相对路径存储"""
    saver = ResultSaver(temp_dir)
    
    # 保存文件
    saver.save_image("aligned.png", np.zeros((600, 600, 3), dtype=np.uint8))
    
    # 在 metadata 中存储相对路径（模拟 Output 对象）
    metadata = {
        "chip_id": "test",
        "aligned_image_path": "aligned.png"  # P1: str, 相对路径
    }
    
    saver.save_json("metadata.json", metadata)
    
    # 加载 metadata
    loaded_meta = saver.load_json("metadata.json")
    
    # 验证路径字段为 str
    assert isinstance(loaded_meta['aligned_image_path'], str)
    assert loaded_meta['aligned_image_path'] == "aligned.png"
    
    # P1: 加载时应能正确转换为绝对路径
    rel_path = loaded_meta['aligned_image_path']
    abs_path = temp_dir / rel_path
    
    assert abs_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
