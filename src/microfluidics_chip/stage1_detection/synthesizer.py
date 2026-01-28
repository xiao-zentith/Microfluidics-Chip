"""
全芯片级物理一致性数据合成器
继承自 v1.0 的 preprocess/synthesizer_chip.py
适配 v1.1 强制规范：
- 使用新的 ChamberDetector 和 CrossGeometryEngine
- 适配 P0 接口（List[ChamberDetection]）
"""

import numpy as np
import cv2
import random
import os
import glob
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from ..core.config import GeometryConfig
from ..core.logger import get_logger
from .detector import ChamberDetector
from .geometry_engine import CrossGeometryEngine

logger = get_logger("stage1_detection.synthesizer")


class FullChipSynthesizer:
    """
    全芯片级物理一致性数据合成器
    
    合成流程：
    1. 信号增强：Masked Random Color Jitter（只改变反应腔）
    2. 物理降质：白平衡漂移 + 光照不均 + 几何变换 + 噪声
    3. 双流切片：Dirty Input / Virtual Clean Label
    """
    
    def __init__(
        self,
        detector: ChamberDetector,
        geometry_config: GeometryConfig,
        class_id_blank: int = 0
    ):
        """
        初始化合成器
        
        :param detector: ChamberDetector 实例
        :param geometry_config: 几何配置
        :param class_id_blank: 空白腔类别ID
        """
        self.detector = detector
        self.geometry_config = geometry_config
        self.class_id_blank = class_id_blank
        
        # 参数配置
        self.params = {
            'color_jitter': (0.6, 1.4),   # 信号层: 模拟不同浓度
            'wb_gain': (0.8, 1.2),        # 光学层: 白平衡漂移
            'illum_strength': 0.7,        # 光学层: 光照不均强度
            'rotation': (-10, 10),        # 几何层: 旋转角度
            'noise_sigma': (0.01, 0.05),  # 传感器层: 噪声
            'mask_radius_ratio': 0.35     # Mask半径占切片大小的比例
        }
        
        logger.info("FullChipSynthesizer initialized")
    
    def _create_concentration_mask(
        self,
        image_shape: tuple,
        detections: List
    ) -> np.ndarray:
        """
        创建局部浓度掩膜
        只覆盖反应腔（Class != BLANK），保持参考腔和背景不变
        
        :param image_shape: 图像形状 (H, W, 3)
        :param detections: ChamberDetection 列表
        :return: mask (H, W, 1) float32
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        if len(detections) == 0:
            return mask[..., np.newaxis]
        
        # 估算半径
        radius = int(self.geometry_config.slice_size[0] * self.params['mask_radius_ratio'])
        
        # 遍历检测结果
        for det in detections:
            # 跳过参考腔（Blank Arm）
            if det.class_id == self.class_id_blank:
                continue
            
            cx, cy = det.center
            cv2.circle(mask, (int(cx), int(cy)), radius, 1.0, -1)
        
        # 高斯模糊：边缘过渡自然
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask[..., np.newaxis]
    
    def _simulate_virtual_concentration(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        层级 1: 信号增强（Masked Random Color Jitter）
        只改变 Mask=1 的区域颜色，模拟浓度变化
        
        :param image: 输入图像 (H, W, 3) float32 [0, 1]
        :param mask: 浓度掩膜 (H, W, 1)
        :return: 增强后的图像
        """
        scale_r = random.uniform(*self.params['color_jitter'])
        scale_g = random.uniform(*self.params['color_jitter'])
        scale_b = random.uniform(*self.params['color_jitter'])
        
        factors = np.ones_like(image)
        
        # BGR 通道
        factors[:, :, 2] = 1.0 + mask[..., 0] * (scale_r - 1.0)  # R
        factors[:, :, 1] = 1.0 + mask[..., 0] * (scale_g - 1.0)  # G
        factors[:, :, 0] = 1.0 + mask[..., 0] * (scale_b - 1.0)  # B
        
        return np.clip(image * factors, 0, 1.0)
    
    def _apply_physics_degradation(self, image: np.ndarray) -> np.ndarray:
        """
        层级 2-4: 物理环境模拟
        - 白平衡漂移
        - 全局光照场
        - 几何变换
        - 传感器噪声
        
        :param image: 输入图像 (H, W, 3) float32 [0, 1]
        :return: 降质后的图像
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
        if radial.max() > 0:
            radial = radial / radial.max()
        
        mix = random.random()
        field = mix * gradient + (1-mix) * radial
        illum_map = 1.0 - self.params['illum_strength'] * field
        out = out * illum_map[..., np.newaxis]
        
        # C. 几何变换
        center = (w // 2, h // 2)
        angle_rot = random.uniform(*self.params['rotation'])
        M = cv2.getRotationMatrix2D(center, angle_rot, 1.0)
        
        # 轻微各向异性缩放
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
    
    def generate_triplets(self, clean_full_img_path: Path) -> List[Dict[str, np.ndarray]]:
        """
        生成单次合成数据对
        
        :param clean_full_img_path: 清洁全图路径
        :return: 三元组列表 [{target_in, ref_in, target_gt}, ...]
        """
        # 1. 读取原始 Clean 图
        img_bgr = cv2.imread(str(clean_full_img_path))
        if img_bgr is None:
            return []
        
        clean_base = img_bgr.astype(np.float32) / 255.0
        
        # 2. 锚定检测（Clean）
        img_uint8_base = (clean_base * 255).astype(np.uint8)
        detections_clean = self.detector.detect(img_uint8_base)
        
        if len(detections_clean) < 12:
            return []
        
        # 3. 生成 Virtual Label Source（虚拟浓度）
        mask = self._create_concentration_mask(clean_base.shape, detections_clean)
        clean_virtual = self._simulate_virtual_concentration(clean_base, mask)
        
        # 4. 生成 Dirty Input Source（物理干扰）
        dirty_input = self._apply_physics_degradation(clean_virtual)
        
        # 5. 双流切片
        # --- 流 A: Dirty Input ---
        dirty_uint8 = (dirty_input * 255).astype(np.uint8)
        detections_dirty = self.detector.detect(dirty_uint8)
        
        # 创建独立引擎
        engine_dirty = CrossGeometryEngine(self.geometry_config)
        _, slices_dirty, _, _ = engine_dirty.process(dirty_uint8, detections_dirty)
        
        if slices_dirty is None or len(slices_dirty) != 12:
            return []
        
        # --- 流 B: Virtual Clean Label ---
        clean_virtual_uint8 = (clean_virtual * 255).astype(np.uint8)
        engine_clean = CrossGeometryEngine(self.geometry_config)
        _, slices_clean, _, _ = engine_clean.process(clean_virtual_uint8, detections_clean)
        
        if slices_clean is None or len(slices_clean) != 12:
            return []
        
        # 6. 组装三元组
        triplets = []
        ref_slice_in = slices_dirty[0].astype(np.float32) / 255.0  # Blank Arm
        
        for i in range(1, 12):
            target_in = slices_dirty[i].astype(np.float32) / 255.0
            target_label = slices_clean[i].astype(np.float32) / 255.0
            
            # 尺寸安全检查
            expected_size = self.geometry_config.slice_size
            if target_in.shape != (expected_size[0], expected_size[1], 3):
                continue
            
            triplets.append({
                'target_in': target_in,
                'ref_in': ref_slice_in,
                'target_gt': target_label
            })
        
        return triplets
    
    def visualize_synthesis_process(
        self, 
        clean_full_img_path: Path, 
        save_path: Path
    ):
        """
        可视化数据增强过程（6面板对比）
        
        展示：
        1. Original Clean - 原始清洁图
        2. Concentration Mask - 浓度掩膜
        3. Virtual Concentration - 虚拟浓度增强
        4. White Balance Drift - 白平衡漂移
        5. Illumination Field - 光照场
        6. Final Dirty Output - 最终干扰输出
        
        :param clean_full_img_path: 清洁全图路径
        :param save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        
        # 读取原始图
        img_bgr = cv2.imread(str(clean_full_img_path))
        if img_bgr is None:
            logger.error(f"Failed to read {clean_full_img_path}")
            return
        
        clean_base = img_bgr.astype(np.float32) / 255.0
        img_uint8 = (clean_base * 255).astype(np.uint8)
        detections = self.detector.detect(img_uint8)
        
        if len(detections) < 12:
            logger.error(f"Insufficient detections: {len(detections)}")
            return
        
        # 生成各阶段图像
        mask = self._create_concentration_mask(clean_base.shape, detections)
        clean_virtual = self._simulate_virtual_concentration(clean_base, mask)
        dirty_final = self._apply_physics_degradation(clean_virtual)
        
        # 转换BGR到RGB用于matplotlib
        clean_rgb = cv2.cvtColor((clean_base * 255).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.0
        virtual_rgb = cv2.cvtColor((clean_virtual * 255).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.0
        dirty_rgb = cv2.cvtColor((dirty_final * 255).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.0
        
        # 绘图（2行3列）
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1
        axes[0, 0].imshow(clean_rgb)
        axes[0, 0].set_title("1. Original Clean", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask.squeeze(), cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title("2. Concentration Mask", fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(virtual_rgb)
        axes[0, 2].set_title("3. Virtual Concentration", fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2 - 差异图
        diff_concentration = np.abs(virtual_rgb - clean_rgb)
        diff_final = np.abs(dirty_rgb - clean_rgb)
        
        axes[1, 0].imshow(diff_concentration)
        axes[1, 0].set_title("4. Concentration Change", fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(dirty_rgb)
        axes[1, 1].set_title("5. Final Dirty Output\n(+WB +Illum +Geo +Noise)", fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(diff_final, vmin=0, vmax=0.5)
        axes[1, 2].set_title("6. Total Degradation", fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Synthesis visualization saved: {save_path}")
    
    def run(
        self,
        clean_dir: Path,
        output_path: Path,
        multiplier: int = 50
    ):
        """
        批量运行合成
        
        :param clean_dir: 清洁图像目录
        :param output_path: 输出 npz 文件路径
        :param multiplier: 每张图像的合成倍率
        """
        all_targets, all_refs, all_labels = [], [], []
        
        files = list(Path(clean_dir).glob("*"))
        files = [f for f in files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        logger.info(f"Starting synthesis: {len(files)} source images, multiplier={multiplier}")
        
        for f in tqdm(files, desc="Synthesizing"):
            for _ in range(multiplier):
                triplets = self.generate_triplets(f)
                for t in triplets:
                    all_targets.append(t['target_in'])
                    all_refs.append(t['ref_in'])
                    all_labels.append(t['target_gt'])
        
        T = np.array(all_targets, dtype=np.float32)
        R = np.array(all_refs, dtype=np.float32)
        L = np.array(all_labels, dtype=np.float32)
        
        logger.info(f"Synthesis complete: {len(T)} slices generated")
        logger.info(f"Target Shape: {T.shape}")
        logger.info(f"Ref Shape:    {R.shape}")
        logger.info(f"Label Shape:  {L.shape}")
        
        np.savez_compressed(output_path, target_in=T, ref_in=R, labels=L)
        logger.info(f"Saved to {output_path}")
