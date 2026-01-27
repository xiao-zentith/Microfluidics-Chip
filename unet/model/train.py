import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random

# 引入我们完善后的模型和Loss
from unet import RefGuidedUNet, ROIWeightedLoss

# ==========================================
# 1. 真实数据集加载器 (npz Loader)
# ==========================================
class MicrofluidicDataset(Dataset):
    def __init__(self, npz_path, mode='train', split_ratio=0.9):
        super().__init__()
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"找不到数据集文件: {npz_path}。请先运行 synthesizer_chip.py")
            
        print(f"[*] Loading data from {npz_path} ...")
        # allow_pickle=True 以防万一
        data = np.load(npz_path, allow_pickle=True)
        
        # 读取数据 (N, 64, 64, 3) -> float32 0-1
        self.t_in = data['target_in']
        self.r_in = data['ref_in']
        self.lbl = data['labels']
        
        total = len(self.t_in)
        split_idx = int(total * split_ratio)
        
        # 划分训练/验证集
        if mode == 'train':
            self.indices = range(0, split_idx)
        else:
            self.indices = range(split_idx, total)
            
        print(f"[{mode.upper()}] Dataset ready: {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # 获取 numpy 数据 (H, W, 3)
        img_sig = self.t_in[real_idx]
        img_ref = self.r_in[real_idx]
        img_gt = self.lbl[real_idx]
        
        # 转换为 PyTorch Tensor 并调整维度 (H, W, 3) -> (3, H, W)
        # 注意: 你的合成数据已经是 float32 (0-1)，不需要再除以 255
        sig_tensor = torch.from_numpy(img_sig).permute(2, 0, 1)
        ref_tensor = torch.from_numpy(img_ref).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(img_gt).permute(2, 0, 1)
        
        return sig_tensor, ref_tensor, gt_tensor

# ==========================================
# 2. 工具函数
# ==========================================
def calculate_psnr(img1, img2):
    """计算 PSNR (dB)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# ==========================================
# 3. 训练引擎
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_pix = 0.0
    running_cos = 0.0
    
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for sig, ref, gt in loop:
        sig, ref, gt = sig.to(device), ref.to(device), gt.to(device)
        
        optimizer.zero_grad()
        output = model(sig, ref)
        
        # 计算混合损失
        loss, l_pix, l_cos = criterion(output, gt)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_pix += l_pix.item()
        running_cos += l_cos.item()
        
        loop.set_postfix(loss=f"{loss.item():.4f}", pix=f"{l_pix.item():.4f}")
        
    count = len(loader)
    return running_loss/count, running_pix/count, running_cos/count

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    
    with torch.no_grad():
        for sig, ref, gt in loader:
            sig, ref, gt = sig.to(device), ref.to(device), gt.to(device)
            output = model(sig, ref)
            loss, _, _ = criterion(output, gt)
            
            val_loss += loss.item()
            val_psnr += calculate_psnr(output, gt).item()
            
    return val_loss / len(loader), val_psnr / len(loader)

# ==========================================
# 4. 结果可视化 (论文级展示)
# ==========================================
def visualize_results(model, dataset, device, epoch, save_dir):
    model.eval()
    # 随机取样
    idx = random.randint(0, len(dataset)-1)
    sig, ref, gt = dataset[idx]
    
    # 推理
    sig_in = sig.unsqueeze(0).to(device)
    ref_in = ref.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sig_in, ref_in)
    
    # 辅助函数: Tensor -> Numpy Image (H, W, 3)
    def to_img(t):
        return t.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    img_in = to_img(sig_in)
    img_out = to_img(output)
    img_gt = to_img(gt)
    
    # 计算差异图 (RGB 距离)
    diff_before = np.sqrt(np.sum((img_in - img_gt)**2, axis=2))
    diff_after = np.sqrt(np.sum((img_out - img_gt)**2, axis=2))
    
    # 绘图
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(img_in)
    axes[0].set_title("Input (Dirty)")
    axes[0].axis('off')
    
    axes[1].imshow(img_out)
    axes[1].set_title("Ours Output")
    axes[1].axis('off')
    
    axes[2].imshow(img_gt)
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    # 差异热力图
    axes[3].imshow(diff_before, cmap='jet', vmin=0, vmax=0.5)
    axes[3].set_title("Error Before")
    axes[3].axis('off')
    
    im = axes[4].imshow(diff_after, cmap='jet', vmin=0, vmax=0.5)
    axes[4].set_title("Error After (Ours)")
    axes[4].axis('off')
    
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    
    path = os.path.join(save_dir, f"epoch_{epoch}_vis.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    # --- 超参数 ---
    NPZ_PATH = "/home/asus515/PycharmProjects/YOLO_v11/preprocess_result/train_data_final.npz" # 你的合成数据
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 100
    SAVE_DIR = "/home/asus515/PycharmProjects/YOLO_v11/unet/result"
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Running on {DEVICE}, saving to {SAVE_DIR}")

    # 1. 数据
    if not os.path.exists(NPZ_PATH):
        print(f"[Error] 请先确保 {NPZ_PATH} 存在！")
        exit()
        
    train_ds = MicrofluidicDataset(NPZ_PATH, mode='train')
    val_ds = MicrofluidicDataset(NPZ_PATH, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. 模型
    model = RefGuidedUNet().to(DEVICE)
    
    # 3. Loss (核心: ROI加权)
    # edge_weight=0.1: 允许边缘有红圈误差
    # roi_radius=20:  对于64x64的图，中间半径20的区域必须算准
    # 不再需要传入 img_size，它会自动识别
    loss_fn = ROIWeightedLoss(roi_radius=20, edge_weight=0.1, lambda_cos=0.2).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    # 4. 训练循环
    best_psnr = 0.0
    
    for epoch in range(1, EPOCHS+1):
        # Train
        t_loss, t_pix, t_cos = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE, epoch)
        
        # Validation
        v_loss, v_psnr = evaluate(model, val_loader, loss_fn, DEVICE)
        
        # Log
        print(f" -> Val Loss: {v_loss:.4f} | Val PSNR: {v_psnr:.2f} dB")
        
        # Save Best
        if v_psnr > best_psnr:
            best_psnr = v_psnr
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(" *** Best Model Saved ***")
            
        # Visualize
        if epoch % 5 == 0:
            visualize_results(model, val_ds, DEVICE, epoch, SAVE_DIR)
            
        # Update LR (monitor PSNR)
        scheduler.step(v_psnr)