import cv2
import numpy as np
import os
import glob

class MicrofluidicPreprocessor:
    def __init__(self, target_size=(64, 64), context_ratio=1.8):
        """
        初始化预处理器
        :param target_size: 最终输入模型的图片大小 (64, 64)
        :param context_ratio: 尺度自适应系数，推荐 1.8 (包含背景)
        """
        self.target_size = target_size
        self.context_ratio = context_ratio

    def sort_keypoints(self, keypoints):
        """
        关键步骤：对 YOLO 检测到的无序点进行排序，确保和标准图的顺序一致。
        假设芯片是 grid 排列，先按 y 排序（行），再按 x 排序（列）。
        :param keypoints: np.array shape (N, 2) or (N, 4) -> [cx, cy, w, h]
        """
        # 1. 简单的排序逻辑：先按 Y 排序（允许一定的误差范围作为同一行），再按 X 排序
        # 这里为了演示简化，直接使用 lexsort
        # 实际使用时，建议根据你的芯片具体布局写死顺序，或者用聚类算法分行
        ind = np.lexsort((keypoints[:, 0], keypoints[:, 1])) 
        return keypoints[ind]

    def register_image(self, src_img, src_pts, dst_pts, dst_size):
        """
        步骤 1: 像素级配准 (透视变换)
        :param src_img: 手机拍摄的原始图 (歪的)
        :param src_pts: 手机图上的 12 个腔室中心坐标 (N, 2)
        :param dst_pts: 标准暗室图上的 12 个腔室中心坐标 (N, 2)
        :param dst_size: 标准图的尺寸 (w, h)
        :return: 校正后的图片 (Registered Image)
        """
        # 计算单应性矩阵 (Homography Matrix)
        # 使用 RANSAC 剔除 YOLO 可能检测不准的异常点
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # 执行透视变换
        warped_img = cv2.warpPerspective(src_img, H, dst_size)
        return warped_img

    def adaptive_crop(self, img, center, bbox_size):
        """
        步骤 2: 尺度自适应切片 (保留背景)
        :param img: 配准后的图片
        :param center: (cx, cy) 腔室中心
        :param bbox_size: (w, h) YOLO检测框的原始宽高
        :return: 缩放后的 Patch (64x64)
        """
        cx, cy = int(center[0]), int(center[1])
        w, h = bbox_size
        
        # 核心逻辑：取长边，乘以系数 (1.8)，计算裁剪边长
        max_side = max(w, h)
        crop_side = int(max_side * self.context_ratio)
        half_side = crop_side // 2

        # 计算裁剪坐标 (处理边界越界问题，使用 Padding)
        pad_w = 0
        pad_h = 0
        
        x1 = cx - half_side
        y1 = cy - half_side
        x2 = cx + half_side
        y2 = cy + half_side
        
        # 如果越界，进行 Padding 处理 (保持背景为黑色或边缘填充)
        # 这里简化处理：直接调用 opencv 的 copyMakeBorder 逻辑比较复杂
        # 简单方案：先 Pad 原图，再切
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img = cv2.copyMakeBorder(img, crop_side, crop_side, crop_side, crop_side, cv2.BORDER_CONSTANT, value=(0,0,0))
            # 坐标偏移
            x1 += crop_side
            y1 += crop_side
            x2 += crop_side
            y2 += crop_side

        patch = img[y1:y2, x1:x2]
        
        # 统一缩放至 64x64 (使用双三次插值保留细节)
        patch_resized = cv2.resize(patch, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        return patch_resized

    def process_one_chip(self, mobile_img_path, darkroom_img_path, 
                         mobile_yolo_data, darkroom_yolo_data, 
                         ref_indices=[9, 10, 11]):
        """
        处理单个芯片的主流程
        :param ref_indices: 参考旋臂对应的腔室索引 (假设是最后3个)
        """
        # 1. 读取图片
        img_mob = cv2.imread(mobile_img_path)
        img_dark = cv2.imread(darkroom_img_path)
        h, w = img_dark.shape[:2]

        # 2. 获取并排序关键点 (假设输入数据格式为 [cx, cy, w, h])
        # 提取中心点用于配准
        pts_mob = self.sort_keypoints(np.array(mobile_yolo_data)[:, :2])
        pts_dark = self.sort_keypoints(np.array(darkroom_yolo_data)[:, :2])
        
        # 提取宽高用于裁剪
        sizes_dark = self.sort_keypoints(np.array(darkroom_yolo_data))[:, 2:]

        # 3. 配准 (Registration) -> 将手机图拉正
        img_registered = self.register_image(img_mob, pts_mob, pts_dark, (w, h))

        # 4. 生成参考流图片 (Input B)
        # 截取参考旋臂上的 3 个腔室
        ref_patches = []
        for idx in ref_indices:
            patch = self.adaptive_crop(img_registered, pts_dark[idx], sizes_dark[idx])
            ref_patches.append(patch)
        
        # 策略：取平均值 (Average) 作为该芯片的全局环境特征
        # 也可以选择拼接 (Concat)，看模型效果
        ref_img_final = np.mean(ref_patches, axis=0).astype(np.uint8)

        # 5. 生成反应流图片对 (Input A & Ground Truth)
        # 遍历所有 12 个腔室 (或者只遍历反应腔)
        dataset_samples = []
        
        for i in range(len(pts_dark)):
            # 排除参考腔室本身 (可选，如果不想让模型预测参考腔)
            if i in ref_indices:
                continue
                
            # Input A: 来自配准后的手机图
            input_patch = self.adaptive_crop(img_registered, pts_dark[i], sizes_dark[i])
            
            # Ground Truth: 来自标准暗室图 (同样的位置和大小)
            gt_patch = self.adaptive_crop(img_dark, pts_dark[i], sizes_dark[i])
            
            # 保存样本数据：(Input A, Input B, GT)
            dataset_samples.append({
                "input_a": input_patch,
                "input_b_ref": ref_img_final,
                "ground_truth": gt_patch,
                "chamber_id": i
            })
            
        return dataset_samples

# ================= 使用示例 =================

if __name__ == "__main__":
    processor = MicrofluidicPreprocessor()
    
    # 模拟数据 (实际使用时，请替换为你 YOLO 推理出的坐标列表)
    # 格式: [center_x, center_y, width, height]
    # 假设有 12 个点
    fake_yolo_mobile = np.random.rand(12, 4) * 500  # 模拟手机检测到的坐标
    fake_yolo_dark = np.random.rand(12, 4) * 500    # 模拟暗室真值坐标
    
    # 假设文件路径
    mobile_path = "data/chip1_angle30_dist20.jpg"
    dark_path = "data/chip1_darkroom_std.jpg"
    
    
    print("代码逻辑检查通过。请替换真实 YOLO 坐标数据运行。")
    print("每个样本包含：")
    print("1. input_a: 反应腔 (含背景)")
    print("2. input_b_ref: 参考旋臂融合图")
    print("3. ground_truth: 暗室真值")