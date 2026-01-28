import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from tqdm import tqdm

# --- 配置参数 ---
HEATMAP_VMAX = 1.0  # 热力图的显示上限。0.5意味着差异超过50%就会显示为最红。
SLICE_IDX_TO_VIS = 6 # 默认查看第6号切片 (通常是中间的反应腔，最具代表性)

def calculate_rgb_diff_map(img1, img2):
    """
    核心算法：计算两张图片的 RGB 欧氏距离差异图
    Args:
        img1, img2: 输入图像 (H, W, 3), 范围可以是 0-255 或 0-1
    Returns:
        diff_map: 差异热力图 (H, W), 范围 0.0 - ~1.73 (sqrt(3))
        mae: 平均绝对误差
    """
    # 1. 统一转换为 float32 且范围归一化到 [0, 1]
    if img1.max() > 1.0: img1 = img1.astype(np.float32) / 255.0
    else: img1 = img1.astype(np.float32)
        
    if img2.max() > 1.0: img2 = img2.astype(np.float32) / 255.0
    else: img2 = img2.astype(np.float32)

    # 2. 计算 RGB 空间的欧氏距离 (Euclidean Distance)
    # 公式: sqrt( (R1-R2)^2 + (G1-G2)^2 + (B1-B2)^2 )
    diff_sq = (img1 - img2) ** 2
    diff_map = np.sqrt(np.sum(diff_sq, axis=2))
    
    # 计算全图平均误差 (用于标题显示)
    mae = np.mean(diff_map)
    
    return diff_map, mae

def visualize_pair(raw_img, gt_img, title_suffix="", save_path=None, show=True):
    """
    绘制对比图: Raw | GT | Difference Heatmap
    """
    diff_map, mae = calculate_rgb_diff_map(raw_img, gt_img)
    
    # 转换颜色空间 BGR -> RGB (用于 matplotlib 显示正常颜色)
    # 假设输入数据保持的是 OpenCV 默认的 BGR 格式
    if raw_img.shape[-1] == 3:
        vis_raw = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) if raw_img.max() > 1 else cv2.cvtColor((raw_img*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        vis_gt = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) if gt_img.max() > 1 else cv2.cvtColor((gt_img*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    else:
        vis_raw, vis_gt = raw_img, gt_img

    # 开始绘图
    plt.figure(figsize=(15, 5))
    
    # 1. Aligned Raw
    plt.subplot(1, 3, 1)
    plt.imshow(vis_raw)
    plt.title(f"Aligned Raw Input\n{title_suffix}", fontsize=11)
    plt.axis('off')
    
    # 2. Aligned GT
    plt.subplot(1, 3, 2)
    plt.imshow(vis_gt)
    plt.title(f"Aligned Ground Truth\nTarget Signal", fontsize=11)
    plt.axis('off')
    
    # 3. RGB Difference Heatmap
    plt.subplot(1, 3, 3)
    # 使用 'jet' 色图: 蓝(无差异) -> 绿 -> 黄 -> 红(大差异)
    im = plt.imshow(diff_map, cmap='jet', vmin=0, vmax=HEATMAP_VMAX)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"RGB Difference Map\nAvg RGB Dist: {mae:.4f}", fontsize=11)
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
        
    if show:
        plt.show()
    
    plt.close()

# ==========================================
# 模式 A: 单个文件读取 (调试用)
# ==========================================
def process_single(npy_path, slice_idx=SLICE_IDX_TO_VIS):
    if not os.path.exists(npy_path):
        print(f"[Error] 文件不存在: {npy_path}")
        return

    print(f"[*] 正在分析单文件: {os.path.basename(npy_path)}")
    data = np.load(npy_path, allow_pickle=True).item()
    
    raw_slices = data['raw_slices']
    gt_slices = data['gt_slices']
    
    if raw_slices[slice_idx] is not None and gt_slices[slice_idx] is not None:
        visualize_pair(
            raw_slices[slice_idx], 
            gt_slices[slice_idx], 
            title_suffix=f"Slice {slice_idx}",
            show=True
        )
    else:
        print(f"[Warn] 该文件的切片 {slice_idx} 为空。")

# ==========================================
# 模式 B: 批量读取 (生成论文素材用)
# ==========================================
def process_batch(input_dir, output_dir, slice_idx=SLICE_IDX_TO_VIS):
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    
    if not npy_files:
        print(f"[Error] 目录 {input_dir} 中未找到 .npy 文件")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"[*] 开始批量处理 {len(npy_files)} 个文件...")
    print(f"[*] 结果将保存至: {output_dir}")
    
    for npy_path in tqdm(npy_files):
        try:
            filename = os.path.basename(npy_path).replace(".npy", "")
            data = np.load(npy_path, allow_pickle=True).item()
            
            raw_s = data['raw_slices'][slice_idx]
            gt_s = data['gt_slices'][slice_idx]
            
            if raw_s is not None and gt_s is not None:
                save_name = os.path.join(output_dir, f"{filename}_slice{slice_idx}_diff.png")
                
                visualize_pair(
                    raw_s, 
                    gt_s, 
                    title_suffix=f"{filename} | Slice {slice_idx}", 
                    save_path=save_name, 
                    show=False # 批量模式下不弹窗
                )
        except Exception as e:
            print(f"[Error] 处理 {filename} 失败: {e}")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # --- 你可以在这里切换模式 ---
    
    # 配置你的数据路径
    DATA_DIR = "/home/asus515/PycharmProjects/YOLO_v11/preprocess_result/processed_dataset"      # 存放 .npy 的文件夹
    RESULT_DIR = "/home/asus515/PycharmProjects/YOLO_v11/visualize_result/analysis_results"  # 存放生成图片的文件夹
    
    # # 1. 模式 A: 随便找一个文件看看效果
    # # 找到第一个文件进行测试
    # sample_files = glob.glob(os.path.join(DATA_DIR, "*.npy"))
    # if sample_files:
    #     print("=== 模式 A: 单样本测试 ===")
    #     process_single(sample_files[0])
    
    # 2. 模式 B: 批量生成所有文件的对比图 (取消注释以运行)
    print("\n=== 模式 B: 批量生成报告 ===")
    process_batch(DATA_DIR, RESULT_DIR)