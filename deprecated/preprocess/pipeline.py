# pipeline.py

import cv2
import numpy as np
import config
from detector import ChamberDetector
from utils import CrossGeometryEngine # 使用新写的类

class PreprocessPipeline:
    def __init__(self):
        self.detector = ChamberDetector()
        self.geo_engine = CrossGeometryEngine() # 初始化几何引擎

    def run(self, raw_img, gt_img, visualize=False):
        # 1. 检测
        pts_raw, cls_raw = self.detector.detect(raw_img)
        pts_gt, cls_gt = self.detector.detect(gt_img)

        # 简单过滤
        if len(pts_raw) != 12 or len(pts_gt) != 12:
            # print(f"[Skip] Point count mismatch: Raw={len(pts_raw)}, GT={len(pts_gt)}")
            return None, None, None

        try:
            # 2. 拓扑排序 (分别处理)
            # 关键：Raw 和 GT 可能旋转角度不同，必须分别 sort_and_anchor
            sorted_raw = self.geo_engine.sort_and_anchor(pts_raw, cls_raw)
            sorted_gt = self.geo_engine.sort_and_anchor(pts_gt, cls_gt)
            
            # 3. 双向变换与切片 (Dual-Warping)
            # Raw -> Ideal
            warped_raw_full, raw_slices = self.geo_engine.dual_warp_and_slice(raw_img, sorted_raw)
            # GT -> Ideal
            warped_gt_full, gt_slices = self.geo_engine.dual_warp_and_slice(gt_img, sorted_gt)
            
            # 4. 可视化 (可选)
            vis_img = None
            if visualize:
                # 拼接显示：左边是 Warped Raw, 右边是 Warped GT
                # 这样你可以直观看到它们是否都变成了“正十字”
                vis_img = np.hstack((warped_raw_full, warped_gt_full))
                                # --- 新增：画出锚点，确认对齐 ---
                # 在左图 (Raw) 的 Ideal Top-Inner 位置画个圈
                cx, cy = self.geo_engine.ideal_points[0] # 第0个点(上臂内侧)
                cv2.circle(vis_img, (int(cx), int(cy)), 10, (0, 0, 255), 3) # 红圈
                
                # 在右图 (GT) 的 Ideal Top-Inner 位置画个圈
                # 右图有 x 偏移 (因为是 hstack)
                offset_x = config.CANVAS_SIZE
                cv2.circle(vis_img, (int(cx + offset_x), int(cy)), 10, (0, 0, 255), 3) # 红圈

                # 缩小一点方便看
                vis_img = cv2.resize(vis_img, (0,0), fx=0.5, fy=0.5)

            return raw_slices, gt_slices, vis_img

        except Exception as e:
            print(f"[Error] Pipeline logic failed: {e}")
            return None, None, None