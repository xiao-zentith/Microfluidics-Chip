import numpy as np
import cv2
import os
import random
from tqdm import tqdm

class SyntheticMicrofluidicGenerator:
    def __init__(self, save_dir="./synthetic_data", img_size=64):
        """
        :param save_dir: 数据保存路径
        :param img_size: 生成图片的尺寸 (默认 64x64，与你切片大小一致)
        """
        self.save_dir = save_dir
        self.img_size = img_size
        self.input_dir = os.path.join(save_dir, "input")
        self.gt_dir = os.path.join(save_dir, "ground_truth")
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.gt_dir, exist_ok=True)

    def generate_gaussian_blob(self, center_x, center_y, sigma, intensity):
        """
        生成一个二维高斯光斑，模拟荧光腔室
        :param intensity: 模拟浓度 (亮度)
        """
        x = np.arange(0, self.img_size, 1, float)
        y = x[:, np.newaxis]
        
        # 生成高斯分布
        x0, y0 = center_x, center_y
        blob = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        
        # 归一化并应用强度
        blob = blob * intensity
        
        # 转为 3 通道 (RGB)，模拟荧光颜色 (通常是绿色或蓝色，这里模拟泛光)
        # 我们可以随机给一点颜色倾向
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        
        # 假设是绿色荧光，G通道最亮
        color_ratio = np.random.uniform(0.8, 1.0, 3) # RGB 随机微调
        img[:, :, 0] = blob * color_ratio[0] * 0.2 # B (少许)
        img[:, :, 1] = blob * color_ratio[1] * 1.0 # G (主色)
        img[:, :, 2] = blob * color_ratio[2] * 0.2 # R (少许)
        
        return np.clip(img, 0, 255)

    def add_lighting_gradient(self, img):
        """
        模拟不均匀光照 (阴阳脸)
        """
        h, w, c = img.shape
        # 随机生成一个梯度蒙版
        gradient_direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        mask = np.zeros((h, w), dtype=np.float32)
        
        if gradient_direction == 'horizontal':
            for i in range(w): mask[:, i] = i / w
        elif gradient_direction == 'vertical':
            for i in range(h): mask[i, :] = i / h
        else:
            for i in range(h):
                for j in range(w):
                    mask[i, j] = (i + j) / (h + w)
        
        # 随机翻转梯度方向
        if random.random() > 0.5:
            mask = 1.0 - mask
            
        # 随机设定光照强度的衰减范围 (比如一边是1.0，一边是0.2)
        min_light = random.uniform(0.1, 0.5)
        max_light = random.uniform(0.8, 1.2)
        mask = mask * (max_light - min_light) + min_light
        
        # 将梯度应用到图像上
        mask = np.expand_dims(mask, axis=-1) # [H, W, 1]
        
        # 增加一点环境底噪 (Ambient Light)，即使是暗处也有光
        ambient_light = np.random.uniform(5, 30) 
        
        return img * mask + ambient_light

    def add_noise_and_blur(self, img):
        """
        模拟相机噪声和失焦
        """
        # 1. 高斯模糊 (模拟对焦不准)
        if random.random() > 0.3:
            k_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k_size, k_size), 0)
            
        # 2. 高斯噪声 (模拟传感器热噪声)
        noise_level = random.uniform(5, 20)
        noise = np.random.normal(0, noise_level, img.shape)
        img = img + noise
        
        return np.clip(img, 0, 255)

    def generate_batch(self, num_samples=1000):
        """
        批量生成数据
        """
        print(f"开始生成 {num_samples} 张合成图像...")
        
        for i in tqdm(range(num_samples)):
            # --- 1. 生成真值 (Ground Truth) ---
            # 随机中心位置 (模拟切片时的轻微偏移)
            cx = self.img_size // 2 + random.randint(-5, 5)
            cy = self.img_size // 2 + random.randint(-5, 5)
            
            # 随机大小 (模拟拍摄距离变化)
            sigma = random.uniform(8, 15)
            
            # 随机亮度 (关键！模拟不同浓度，范围 50-250)
            # 这一步是为了解决你"单一浓度"的隐患
            intensity = random.uniform(50, 250)
            
            gt_img = self.generate_gaussian_blob(cx, cy, sigma, intensity)
            
            # --- 2. 生成输入 (Input) ---
            # 复制一份真值开始加干扰
            input_img = gt_img.copy()
            
            # 加上光照梯度 (模拟参考旋臂和反应腔光照不一致)
            input_img = self.add_lighting_gradient(input_img)
            
            # 加上噪声和模糊
            input_img = self.add_noise_and_blur(input_img)
            
            # --- 3. 保存 ---
            # 转换为 uint8 格式保存
            gt_save = gt_img.astype(np.uint8)
            input_save = input_img.astype(np.uint8)
            
            cv2.imwrite(os.path.join(self.gt_dir, f"syn_{i:05d}.jpg"), gt_save)
            cv2.imwrite(os.path.join(self.input_dir, f"syn_{i:05d}.jpg"), input_save)
            
        print(f"生成完毕！数据保存在: {self.save_dir}")

# ================= 运行脚本 =================
if __name__ == "__main__":
    # 生成 5000 张用于预训练
    generator = SyntheticMicrofluidicGenerator(save_dir="./synthetic_data_v1")
    generator.generate_batch(num_samples=5000)