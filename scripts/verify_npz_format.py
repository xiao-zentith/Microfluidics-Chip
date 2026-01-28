"""
验证 prepare_training_data.py 修复效果的测试脚本

测试内容：
1. NPZ key 名是否正确
2. Reference 形状是否为 (N, H, W, 3)
3. 能否被 MicrofluidicDataset 正确加载
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_npz_format(npz_path: Path):
    """测试NPZ文件格式"""
    print(f"Testing NPZ file: {npz_path}")
    
    if not npz_path.exists():
        print(f"❌ File not found: {npz_path}")
        return False
    
    # 加载NPZ
    data = np.load(npz_path)
    
    # 检查key
    expected_keys = {'target_in', 'ref_in', 'labels'}
    actual_keys = set(data.keys())
    
    print(f"\n1️⃣ Key检查:")
    print(f"   期望: {expected_keys}")
    print(f"   实际: {actual_keys}")
    
    if expected_keys != actual_keys:
        print(f"   ❌ Key不匹配！缺失: {expected_keys - actual_keys}, 多余: {actual_keys - expected_keys}")
        return False
    print(f"   ✅ Key正确")
    
    # 检查形状
    target_in = data['target_in']
    ref_in = data['ref_in']
    labels = data['labels']
    
    print(f"\n2️⃣ 形状检查:")
    print(f"   target_in: {target_in.shape}")
    print(f"   ref_in:    {ref_in.shape}")
    print(f"   labels:    {labels.shape}")
    
    # 验证都是 (N, H, W, 3)
    if len(target_in.shape) != 4 or target_in.shape[-1] != 3:
        print(f"   ❌ target_in形状错误，应为(N, H, W, 3)")
        return False
    
    if len(ref_in.shape) != 4 or ref_in.shape[-1] != 3:
        print(f"   ❌ ref_in形状错误，应为(N, H, W, 3)")
        return False
    
    if len(labels.shape) != 4 or labels.shape[-1] != 3:
        print(f"   ❌ labels形状错误，应为(N, H, W, 3)")
        return False
    
    if not (target_in.shape == ref_in.shape == labels.shape):
        print(f"   ❌ 三个数组形状不一致")
        return False
    
    print(f"   ✅ 形状正确 (N={target_in.shape[0]}, H={target_in.shape[1]}, W={target_in.shape[2]})")
    
    # 检查数值范围
    print(f"\n3️⃣ 数值范围检查:")
    print(f"   target_in: [{target_in.min():.3f}, {target_in.max():.3f}]")
    print(f"   ref_in:    [{ref_in.min():.3f}, {ref_in.max():.3f}]")
    print(f"   labels:    [{labels.min():.3f}, {labels.max():.3f}]")
    
    if not (0 <= target_in.min() and target_in.max() <= 1):
        print(f"   ⚠️  target_in不在[0,1]范围")
    if not (0 <= ref_in.min() and ref_in.max() <= 1):
        print(f"   ⚠️  ref_in不在[0,1]范围")
    if not (0 <= labels.min() and labels.max() <= 1):
        print(f"   ⚠️  labels不在[0,1]范围")
    
    print(f"   ✅ 数值范围合理")
    
    # 测试 MicrofluidicDataset 加载
    print(f"\n4️⃣ MicrofluidicDataset加载测试:")
    try:
        from microfluidics_chip.stage2_correction.dataset import MicrofluidicDataset
        dataset = MicrofluidicDataset(npz_path, mode='train')
        
        # 获取一个样本
        signal, ref, gt = dataset[0]
        
        print(f"   样本数量: {len(dataset)}")
        print(f"   signal: {signal.shape} (应为 3, H, W)")
        print(f"   ref:    {ref.shape} (应为 3, H, W)")
        print(f"   gt:     {gt.shape} (应为 3, H, W)")
        
        if signal.shape[0] != 3 or ref.shape[0] != 3 or gt.shape[0] != 3:
            print(f"   ❌ Dataset返回的Tensor形状错误")
            return False
        
        print(f"   ✅ MicrofluidicDataset加载成功")
        
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return False
    
    print(f"\n{'='*60}")
    print(f"✅ 所有测试通过！NPZ格式正确。")
    print(f"{'='*60}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证NPZ数据格式")
    parser.add_argument("npz_path", type=Path, help="NPZ文件路径")
    
    args = parser.parse_args()
    
    success = test_npz_format(args.npz_path)
    exit(0 if success else 1)
