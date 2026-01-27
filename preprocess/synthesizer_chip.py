import numpy as np
import cv2
import random
import os
import glob
from tqdm import tqdm
import config

# === 导入你提供的真实模块 ===
from utils import CrossGeometryEngine
from detector import ChamberDetector  # 修正类名

class FullChipSynthesizer:
    """
    全芯片级物理一致性数据合成器 (适配 V10.0 utils 和 detector)
    """
    
    def __init__(self):
        # 实例化检测器和几何引擎
        self.detector = ChamberDetector()
        self.engine = CrossGeometryEngine()
        
        # 参数配置
        self.params = {
            'color_jitter': (0.6, 1.4),   # 信号层: 模拟不同浓度 (RGB独立缩放)
            'wb_gain': (0.8, 1.2),        # 光学层: 白平衡漂移
            'illum_strength': 0.7,        # 光学层: 光照不均强度
            'rotation': (-10, 10),        # 几何层: 旋转角度
            'noise_sigma': (0.01, 0.05),  # 传感器层: 噪声
            'mask_radius_ratio': 0.35     # Mask半径占切片大小的比例
        }

    def _create_concentration_mask(self, image_shape, points, classes):
        """
        创建局部浓度掩膜。
        只覆盖反应腔 (Class != BLANK)，保持参考腔和背景不变。
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        if points is None or len(points) == 0:
            return mask[..., np.newaxis]

        # 估算半径: 使用 config.SLICE_SIZE 的一定比例
        # 因为 detector 不返回宽高，我们假设切片大小能涵盖腔室
        radius = int(config.SLICE_SIZE[0] * self.params['mask_radius_ratio'])

        # 遍历检测点和类别
        for (cx, cy), cls_id in zip(points, classes):
            # !!! 关键: 跳过参考腔 (Blank Arm) !!!
            # 参考腔的浓度不随待测液变化，必须保持黑色 (0)
            if cls_id == config.CLASS_ID_BLANK:
                continue
            
            # 画实心白圆
            cv2.circle(mask, (int(cx), int(cy)), radius, 1.0, -1)
            
        # 高斯模糊：让边缘过渡自然
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask[..., np.newaxis]

    def _simulate_virtual_concentration(self, image, mask):
        """
        层级 1: 信号增强 (Masked Random Color Jitter)
        只改变 Mask=1 的区域颜色，模拟浓度变化。
        """
        scale_r = random.uniform(*self.params['color_jitter'])
        scale_g = random.uniform(*self.params['color_jitter'])
        scale_b = random.uniform(*self.params['color_jitter'])
        
        factors = np.ones_like(image)
        
        # 假设 image 是 BGR (OpenCV 默认)
        factors[:, :, 2] = 1.0 + mask[..., 0] * (scale_r - 1.0) # R
        factors[:, :, 1] = 1.0 + mask[..., 0] * (scale_g - 1.0) # G
        factors[:, :, 0] = 1.0 + mask[..., 0] * (scale_b - 1.0) # B
        
        return np.clip(image * factors, 0, 1.0)

    def _apply_physics_degradation(self, image):
        """
        层级 2-4: 物理环境模拟 (光照 + 白平衡 + 几何 + 噪声)
        """
        h, w = image.shape[:2]
        out = image.copy()
        
        # A. 白平衡漂移
        r_gain = random.uniform(*self.params['wb_gain'])
        b_gain = random.uniform(*self.params['wb_gain'])
        out[:, :, 2] *= r_gain
        out[:, :, 0] *= b_gain
        
        # B. 全局光照场
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        angle = np.deg2rad(random.uniform(0, 360))
        gradient = X * np.cos(angle) + Y * np.sin(angle)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
        
        cx, cy = w // 2, h // 2
        radial = np.sqrt((X - cx)**2 + (Y - cy)**2)
        if radial.max() > 0: radial = radial / radial.max()
        
        mix = random.random()
        field = mix * gradient + (1-mix) * radial
        illum_map = 1.0 - self.params['illum_strength'] * field
        out = out * illum_map[..., np.newaxis]
        
        # C. 几何变换 (旋转 + 缩放模拟)
        center = (w // 2, h // 2)
        angle_rot = random.uniform(*self.params['rotation'])
        M = cv2.getRotationMatrix2D(center, angle_rot, 1.0)
        
        # 轻微各向异性缩放 (模拟透视/椭圆)
        scale_x = random.uniform(0.95, 1.05)
        scale_y = random.uniform(0.95, 1.05)
        M[0, 0] *= scale_x
        M[1, 1] *= scale_y
        
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        # D. 传感器噪声
        sigma = random.uniform(*self.params['noise_sigma'])
        noise = np.random.normal(0, sigma, out.shape)
        out = out + noise
        
        return np.clip(out, 0, 1.0)

    def generate_triplets(self, clean_full_img_path):
        """
        生成单次合成数据对
        """
        # 1. 读取原始 Clean 图
        img_bgr = cv2.imread(clean_full_img_path)
        if img_bgr is None: return []
        clean_base = img_bgr.astype(np.float32) / 255.0
        
        # === Step 1: 锚定检测 (Clean) ===
        # 调用 detector 获取点和类别
        img_uint8_base = (clean_base * 255).astype(np.uint8)
        pts_clean, cls_clean = self.detector.detect(img_uint8_base)
        
        # 校验: 点数不足跳过
        if len(pts_clean) < 12:
            return []

        # === Step 2: 生成 Virtual Label Source (虚拟浓度) ===
        # 使用检测到的点和类别画 Mask
        mask = self._create_concentration_mask(clean_base.shape, pts_clean, cls_clean)
        
        # 得到虚拟真值图 (背景不变，反应腔变色)
        clean_virtual = self._simulate_virtual_concentration(clean_base, mask)
        
        # === Step 3: 生成 Dirty Input Source (物理干扰) ===
        # 对虚拟真值图进行破坏
        dirty_input = self._apply_physics_degradation(clean_virtual)
        
        # === Step 4: 双流切片 (Dual-Stream Slicing) ===
        
        # --- 流 A: 处理 Input (Dirty) ---
        # 1. 重新检测! (模拟真实抖动)
        dirty_uint8 = (dirty_input * 255).astype(np.uint8)
        pts_dirty, cls_dirty = self.detector.detect(dirty_uint8)
        
        # 2. 调用 Engine 进行处理 (传入检测结果)
        # process 返回 (patches, debug_vis)
        slices_dirty, _ = self.engine.process(dirty_uint8, pts_dirty, cls_dirty)
        
        if slices_dirty is None: # 比如检测点太少被 process 拒了
            return []

        # --- 流 B: 处理 Label (Virtual Clean) ---
        # 使用 Step 1 的干净图检测结果进行切片
        # 这样保证 Label 是基于理想坐标切出来的
        clean_virtual_uint8 = (clean_virtual * 255).astype(np.uint8)
        slices_clean, _ = self.engine.process(clean_virtual_uint8, pts_clean, cls_clean)

        if slices_clean is None:
            return []

        # 校验切片数量 (必须都是 12 个)
        # engine.process 内部通过拓扑排序保证了返回的 patches 是 0-11 顺序
        if len(slices_dirty) != 12 or len(slices_clean) != 12:
            return []

        # === Step 5: 组装三元组 ===
        triplets = []
        
        # 提取 Reference Input (Dirty Stream 的第 0 个 - Blank Arm)
        ref_slice_in = slices_dirty[0].astype(np.float32) / 255.0
        
        # 遍历 Target (1-11)
        for i in range(1, 12):
            target_in = slices_dirty[i].astype(np.float32) / 255.0
            target_label = slices_clean[i].astype(np.float32) / 255.0
            
            # 尺寸安全检查
            if target_in.shape != (config.SLICE_SIZE[0], config.SLICE_SIZE[1], 3):
                continue
            
            triplets.append({
                'target_in': target_in,   # 输入: 脏、偏色、抖动
                'ref_in': ref_slice_in,   # 参考: 脏、同源光照
                'target_gt': target_label # 标签: 净、虚拟浓度
            })
            
        return triplets

    def run(self, clean_dir, output_path, multiplier=50):
        """批量运行"""
        all_targets, all_refs, all_labels = [], [], []
        
        files = glob.glob(os.path.join(clean_dir, "*"))
        print(f"[*] 开始合成，源图像: {len(files)} 张，倍率: {multiplier}")
        
        for f in tqdm(files):
            for _ in range(multiplier):
                triplets = self.generate_triplets(f)
                for t in triplets:
                    all_targets.append(t['target_in'])
                    all_refs.append(t['ref_in'])
                    all_labels.append(t['target_gt'])
                    
        T = np.array(all_targets, dtype=np.float32)
        R = np.array(all_refs, dtype=np.float32)
        L = np.array(all_labels, dtype=np.float32)
        
        print(f"[完成] 生成切片总数: {len(T)}")
        print(f"Target Shape: {T.shape}")
        print(f"Ref Shape:    {R.shape}")
        print(f"Label Shape:  {L.shape}")
        
        np.savez_compressed(output_path, target_in=T, ref_in=R, labels=L)
        print(f"[Saved] {output_path}")

if __name__ == "__main__":
    CLEAN_DIR = "/home/asus515/PycharmProjects/YOLO_v11/dataset/clean_full_images"
    OUTPUT_FILE = "/home/asus515/PycharmProjects/YOLO_v11/preprocess_result/train_data_final.npz"
    
    if not os.path.exists(CLEAN_DIR):
        print(f"请创建文件夹 {CLEAN_DIR} 并放入标准全图")
    else:
        syn = FullChipSynthesizer()
        syn.run(CLEAN_DIR, OUTPUT_FILE, multiplier=100)