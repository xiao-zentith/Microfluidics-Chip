import numpy as np
import cv2
import random
import glob
import os
from tqdm import tqdm
import config

class MicrofluidicSynthesizer:
    """
    微流控芯片荧光图像数据合成器 (RGB版)
    目标：输入干净的 GT (H,W,3)，输出模拟干扰的 Input (H,W,3)。
    严格保留色彩通道信息，用于浓度回归任务。
    """

    def __init__(self, debug_mode=False):
        self.slice_size = config.SLICE_SIZE[0]
        self.debug = debug_mode
        self.params = {
            'intensity_range': (0.6, 1.2),      # 全局亮度增益
            'illumination_strength': 0.4,       # 光照不均强度
            'noise_sigma': (0.01, 0.05),        # 传感器噪声
            'geo_scale': (0.9, 1.1),
            'geo_angle': (-5, 5),
            'stretch_ratio': (0.95, 1.05)
        }

    def _apply_geometry(self, image, is_label=False):
        """
        几何变换 (支持 3通道)
        """
        # [核心修复] 使用切片获取 H, W，忽略通道数 C
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        angle = random.uniform(*self.params['geo_angle'])
        scale = random.uniform(*self.params['geo_scale'])
        
        stretch_x = random.uniform(*self.params['stretch_ratio'])
        stretch_y = random.uniform(*self.params['stretch_ratio'])
        
        M_rot = cv2.getRotationMatrix2D(center, angle, scale)
        
        if not is_label:
            M_rot[0, 0] *= stretch_x
            M_rot[1, 1] *= stretch_y

        # OpenCV 的 warpAffine 自动处理多通道图像
        # borderMode=cv2.BORDER_REFLECT 保证边缘填充自然，不引入全黑边界
        warped = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return warped

    def _simulate_uneven_illumination(self, image):
        """
        模拟光照不均 (作用于所有通道)
        """
        h, w = image.shape[:2]
        
        mode = random.choice(['linear', 'radial'])
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        
        if mode == 'linear':
            angle = np.deg2rad(random.uniform(0, 360))
            gradient = X * np.cos(angle) + Y * np.sin(angle)
            gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
            field = 1.0 - self.params['illumination_strength'] * (1 - gradient)
        else: # radial
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            sigma = random.uniform(w*0.5, w*1.5)
            gauss = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
            field = 1.0 - self.params['illumination_strength'] * (1 - gauss)

        # [核心逻辑] 光照场是单通道的 (H,W)，图像是 (H,W,3)
        # 需要将光照场扩展维度为 (H,W,1) 以进行广播乘法
        field = field[..., np.newaxis]
        
        return image * field

    def _add_noise(self, image):
        """
        模拟成像噪声 (通道独立)
        """
        # 生成与图像 shape 完全一致的噪声 (H, W, 3)
        # 这模拟了 RGB 传感器上每个像素点、每个滤光片下的独立热噪
        sigma = random.uniform(*self.params['noise_sigma'])
        noise = np.random.normal(0, sigma, image.shape)
        
        noisy_img = image + noise
        
        # 全局亮度波动 (模拟曝光时间/模拟增益的变化，作用于所有通道)
        brightness = random.uniform(*self.params['intensity_range'])
        noisy_img = noisy_img * brightness
        
        return np.clip(noisy_img, 0, 1)

    def synthesize_pair(self, clean_gt):
        """
        执行合成
        Input clean_gt: (H, W, 3) 
        """
        # 1. 归一化输入 (确保是 float32 0-1)
        # 即使是整数，除以255后仍保留三通道比例关系
        if clean_gt.max() > 1.0:
            clean_gt = clean_gt.astype(np.float32) / 255.0
        else:
            clean_gt = clean_gt.astype(np.float32)

        # [安全检查] 确保是三维数据，如果是灰度图强行升维，防止广播错误
        if clean_gt.ndim == 2:
            clean_gt = np.expand_dims(clean_gt, axis=2) # (H,W) -> (H,W,1)
            # 如果你的项目必须是3通道（例如后续网络输入层是3），这里可能需要 repeat
            # clean_gt = np.repeat(clean_gt, 3, axis=2) 

        # 2. 几何增强 (Input/Label 同步或轻微差异)
        aug_gt = self._apply_geometry(clean_gt, is_label=True)
        aug_input = aug_gt.copy()
        
        # 3. 物理干扰 (仅作用于 Input，且保持 RGB 关系)
        aug_input = self._simulate_uneven_illumination(aug_input)
        aug_input = self._add_noise(aug_input)
        
        return aug_input.astype(np.float32), aug_gt.astype(np.float32)

    def process_dataset(self, npy_paths, output_path, multiplier=5):
        all_inputs = []
        all_labels = []
        
        print(f"[*] 开始数据合成 (RGB模式)，倍率: {multiplier}x")
        
        for npy_file in npy_paths:
            data = np.load(npy_file, allow_pickle=True).item()
            gt_slices = data['gt_slices'] 
            
            valid_gts = [s for s in gt_slices if s is not None]
            
            for gt in tqdm(valid_gts, desc=f"Processing {os.path.basename(npy_file)}"):
                for _ in range(multiplier):
                    syn_in, syn_label = self.synthesize_pair(gt)
                    all_inputs.append(syn_in)
                    all_labels.append(syn_label)
        
        X = np.array(all_inputs).astype(np.float32)
        Y = np.array(all_labels).astype(np.float32)
        
        # 此时 X, Y 的 shape 应该是 (N, H, W, C)，通常 C=3
        print(f"[OK] 合成完成。")
        print(f"    Inputs Shape: {X.shape}")
        print(f"    Labels Shape: {Y.shape}")
        
        np.savez_compressed(output_path, inputs=X, labels=Y)
        print(f"[Save] 数据集已保存至: {output_path}")

if __name__ == "__main__":
    # 测试代码
    pass