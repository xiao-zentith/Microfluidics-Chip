import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

def visualize_generated_data(npz_path, num_samples=5):
    """
    可视化 .npz 数据集中的样本
    """
    if not os.path.exists(npz_path):
        print(f"[Error] 文件未找到: {npz_path}")
        print("请先运行 synthesizer_chip.py 生成数据。")
        return

    print(f"[*] 正在加载 {npz_path} ...")
    data = np.load(npz_path)
    
    # 获取数组
    targets_in = data['target_in']  # (N, 64, 64, 3)
    refs_in = data['ref_in']        # (N, 64, 64, 3)
    labels = data['labels']         # (N, 64, 64, 3)
    
    total_samples = len(targets_in)
    print(f"[*] 数据集包含 {total_samples} 个样本。正在随机抽取 {num_samples} 个进行展示...")

    # 设置绘图布局: 行=样本数, 列=4 (Target_In, Ref_In, Label, Diff)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    # 随机索引
    indices = random.sample(range(total_samples), num_samples)

    for row_idx, i in enumerate(indices):
        # 1. 获取数据 (注意：假设数据是 float32 0-1)
        t_in = targets_in[i]
        r_in = refs_in[i]
        lbl = labels[i]
        
        # 2. 格式转换 (BGR -> RGB)
        # 合成器用 cv2 读取，所以 npz 里存的是 BGR。
        # matplotlib 显示需要 RGB。
        t_in_rgb = cv2.cvtColor(t_in, cv2.COLOR_BGR2RGB)
        r_in_rgb = cv2.cvtColor(r_in, cv2.COLOR_BGR2RGB)
        lbl_rgb = cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB)
        
        # 3. 计算差异热力图 (RGB 欧氏距离)
        # 展示 Input 和 Label 的差距
        diff = np.sqrt(np.sum((t_in - lbl) ** 2, axis=2))
        
        # --- 绘图 ---
        ax = axes[row_idx] if num_samples > 1 else axes
        
        # Col 1: Target Input (Input Stream A)
        ax[0].imshow(t_in_rgb)
        if row_idx == 0: ax[0].set_title("Target Input\n(Dirty/Augmented)", fontsize=12, color='red')
        ax[0].axis('off')
        
        # Col 2: Reference Input (Input Stream B)
        ax[1].imshow(r_in_rgb)
        if row_idx == 0: ax[1].set_title("Reference Input\n(Context/Blank)", fontsize=12, color='blue')
        ax[1].axis('off')
        
        # Col 3: Target Label (Ground Truth)
        ax[2].imshow(lbl_rgb)
        if row_idx == 0: ax[2].set_title("Target Label\n(Virtual GT)", fontsize=12, color='green')
        ax[2].axis('off')
        
        # Col 4: Difference Map
        im = ax[3].imshow(diff, cmap='jet', vmin=0, vmax=0.6)
        if row_idx == 0: ax[3].set_title("Difference Map\n(What to learn)", fontsize=12)
        ax[3].axis('off')
        
        # 在左侧标注样本索引
        ax[0].text(-10, 32, f"Sample\n#{i}", fontsize=10, ha='right', va='center')

    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='RGB Difference')
    
    plt.suptitle(f"Data Augmentation Preview: {os.path.basename(npz_path)}", fontsize=16)
    
    save_path = "augmentation_preview.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Done] 预览图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    NPZ_FILE = "/home/asus515/PycharmProjects/YOLO_v11/preprocess_result/train_data_final.npz" # 请确保文件名与 synthesizer_chip.py 输出一致
    visualize_generated_data(NPZ_FILE, num_samples=5)