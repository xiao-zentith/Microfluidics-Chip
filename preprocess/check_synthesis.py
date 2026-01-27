import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random
import cv2
from synthesizer import MicrofluidicSynthesizer
import config

def visualize_synthesis(data_dir="/home/asus515/PycharmProjects/YOLO_v11/preprocess_result/processed_dataset", num_samples=5):
    """
    可视化合成效果：对比 原始GT vs 合成Input vs 合成Label
    """
    # 1. 初始化合成器
    syn = MicrofluidicSynthesizer(debug_mode=True)
    
    # 2. 获取数据源
    npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not npy_files:
        print(f"[Error] 目录 {data_dir} 下未找到 .npy 文件。")
        return

    # 3. 设置绘图画布
    # 布局：每行显示一个样本，三列分别为 [原始GT] -> [合成Input (脏)] -> [合成Label (净)]
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    if num_samples == 1: axes = [axes] # 处理单样本时的维度问题

    print(f"[*] 正在随机抽取 {num_samples} 个样本进行合成测试...")

    for i in range(num_samples):
        # 随机读取文件和切片
        npy_path = random.choice(npy_files)
        data = np.load(npy_path, allow_pickle=True).item()
        gt_slices = [s for s in data['gt_slices'] if s is not None]
        
        if not gt_slices:
            continue
            
        # 拿到原始纯净切片
        clean_gt = random.choice(gt_slices)
        
        # *** 核心：调用合成器 ***
        # synth_input: 模拟手机拍摄 (噪声、光照、形变)
        # synth_label: 训练目标 (仅形变，无噪声)
        synth_input, synth_label = syn.synthesize_pair(clean_gt)
        
        # --- 绘图逻辑 ---
        
        # Column 1: Original Source (原始库中的切片)
        ax_org = axes[i][0] if num_samples > 1 else axes[0]
        ax_org.imshow(clean_gt, cmap='gray', vmin=0, vmax=255)
        ax_org.set_title("Source GT (Raw Library)", fontsize=10)
        ax_org.axis('off')
        
        # Column 2: Synthetic Input (模型输入)
        # 叠加了光照、噪声、几何变换
        ax_in = axes[i][1] if num_samples > 1 else axes[1]
        ax_in.imshow(synth_input.squeeze(), cmap='gray', vmin=0, vmax=1.0)
        ax_in.set_title(f"Synthetic Input\n(Noise+Light+Transform)", fontsize=10, color='red')
        ax_in.axis('off')
        
        # Column 3: Synthetic Label (模型真值)
        # 应该清晰、无噪，但几何形态与 Input 必须严格对齐
        ax_label = axes[i][2] if num_samples > 1 else axes[2]
        ax_label.imshow(synth_label.squeeze(), cmap='gray', vmin=0, vmax=1.0)
        ax_label.set_title(f"Target Label\n(Clean+Aligned)", fontsize=10, color='green')
        ax_label.axis('off')

    # 保存对比图到本地，方便查看
    save_path = "synthesis_verification_report.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Success] 可视化报告已生成: {save_path}")
    print("请打开图片检查：Input 是否足够'脏'？Label 是否足够'净'？两者几何是否'对齐'？")
    plt.show()

if __name__ == "__main__":
    visualize_synthesis(num_samples=4)