import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def check_data(npz_path):
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    signals = data['target_in']
    targets = data['labels']
    refs = data['ref_in']
    
    print(f"Signals: {signals.shape}, dtype={signals.dtype}")
    print(f"Targets: {targets.shape}, dtype={targets.dtype}")
    
    # 1. 统计信息
    print("\nStatistics:")
    print(f"Signal: min={signals.min():.4f}, max={signals.max():.4f}, mean={signals.mean():.4f}")
    print(f"Target: min={targets.min():.4f}, max={targets.max():.4f}, mean={targets.mean():.4f}")
    
    # 2. 对齐检查 (取第一个样本)
    idx = 0
    sig = signals[idx]
    tgt = targets[idx]
    
    # 转换为 uint8 用于显示
    sig_u8 = (np.clip(sig, 0, 1) * 255).astype(np.uint8)
    tgt_u8 = (np.clip(tgt, 0, 1) * 255).astype(np.uint8)
    
    # 创建叠加图 (Signal=绿, Target=紫)
    overlay = cv2.addWeighted(sig_u8, 0.5, tgt_u8, 0.5, 0)
    
    # 3. 可视化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("Signal (Input)")
    plt.imshow(cv2.cvtColor(sig_u8, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 4, 2)
    plt.title("Target (GT)")
    plt.imshow(cv2.cvtColor(tgt_u8, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 4, 3)
    plt.title("Overlay (Check Alignment)")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 4, 4)
    plt.title("Difference (|Sig - Tgt|)")
    diff = cv2.absdiff(sig_u8, tgt_u8)
    plt.imshow(diff, cmap='gray')
    
    out_path = "debug_data_check.png"
    plt.savefig(out_path)
    print(f"\nVisualization saved to {out_path}")
    print("Please check 'Overlay' and 'Difference' for misalignment.")

if __name__ == "__main__":
    check_data("processed_data/training.npz")
