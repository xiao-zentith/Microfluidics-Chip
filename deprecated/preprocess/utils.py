import cv2
import numpy as np
import math
import config

class CrossGeometryEngine:
    """
    V10.0 最终完善版
    1. 真实坐标跟随 (Real-Coordinate Following)：切片精准居中。
    2. 身份跟随 (Identity Following)：红圈基于 Class ID 绘制，绝不错位。
    3. 重心排序 (Centroid Sorting)：基于芯片重心排序，解决排序倒置问题。
    """
    def __init__(self):
        self.center = (config.CANVAS_SIZE // 2, config.CANVAS_SIZE // 2)
        self.ideal_slice_centers = self._generate_ideal_centers()

    def _generate_ideal_centers(self):
        centers = []
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)] # N, E, S, W
        for dx, dy in directions:
            for i in range(3):
                dist = config.IDEAL_CENTER_GAP + i * config.IDEAL_CHAMBER_STEP
                cx = self.center[0] + dx * dist
                cy = self.center[1] + dy * dist
                centers.append((cx, cy))
        return centers

    def _get_manual_rigid_matrix(self, arms, blank_arm_idx, src_centroid):
        """计算刚性变换矩阵"""
        # 使用传入的 src_centroid (芯片重心)
        src_cx, src_cy = src_centroid
        
        # Source Angle (Blank Arm)
        blank_pts = [p['pt'] for p in arms[blank_arm_idx]]
        b_avg_x = np.mean([p[0] for p in blank_pts])
        b_avg_y = np.mean([p[1] for p in blank_pts])
        src_angle_rad = math.atan2(b_avg_y - src_cy, b_avg_x - src_cx)
        
        # Source Scale (基于到芯片重心的距离)
        all_pts = []
        for pts in arms.values():
            for p in pts:
                all_pts.append(p['pt'])
        dists = [math.sqrt((p[0]-src_cx)**2 + (p[1]-src_cy)**2) for p in all_pts]
        src_scale = np.mean(dists)

        # Target Params
        dst_cx, dst_cy = self.center
        dst_angle_rad = math.radians(-90) # Top
        dst_scale = np.mean([
            config.IDEAL_CENTER_GAP,
            config.IDEAL_CENTER_GAP + config.IDEAL_CHAMBER_STEP,
            config.IDEAL_CENTER_GAP + 2 * config.IDEAL_CHAMBER_STEP
        ])

        # Delta
        delta_angle = dst_angle_rad - src_angle_rad
        scale_factor = dst_scale / src_scale
        
        # Matrix Construction
        T1 = np.array([[1, 0, -src_cx], [0, 1, -src_cy], [0, 0, 1]])
        c, s = math.cos(delta_angle), math.sin(delta_angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        S = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
        T2 = np.array([[1, 0, dst_cx], [0, 1, dst_cy], [0, 0, 1]])
        
        M_3x3 = T2 @ S @ R @ T1
        return M_3x3[:2, :]

    def process(self, img, keypoints, classes):
        if len(keypoints) < 12:
            return None, None

        h, w = img.shape[:2]
        
        # --- Step 0: 计算芯片重心 (用于稳健排序) ---
        kp_np = np.array(keypoints)
        chip_cx = np.mean(kp_np[:, 0])
        chip_cy = np.mean(kp_np[:, 1])

        # --- Step 1: 拓扑排序 ---
        polar_points = []
        for i, (x, y) in enumerate(keypoints):
            # 基于芯片重心计算极坐标，而非图片中心
            dx, dy = x - chip_cx, y - chip_cy
            angle = math.degrees(math.atan2(dy, dx)) % 360
            dist = math.sqrt(dx**2 + dy**2)
            polar_points.append({'angle': angle, 'dist': dist, 'cls': classes[i], 'pt': (x, y)})

        polar_points.sort(key=lambda p: p['angle'])

        angles = [p['angle'] for p in polar_points]
        gaps = []
        for i in range(len(angles)):
            diff = (angles[(i + 1) % len(angles)] - angles[i]) % 360
            gaps.append((diff, i))
        
        gaps.sort(key=lambda x: x[0], reverse=True)
        top_4_indices = sorted([g[1] for g in gaps[:4]])

        arms = {}
        start_idx = (top_4_indices[-1] + 1) % len(polar_points)
        sorted_indices = list(range(len(polar_points)))
        sorted_indices = sorted_indices[start_idx:] + sorted_indices[:start_idx]
        
        gap_set = set(top_4_indices)
        current_pts = []
        arm_id = 0
        
        for i in sorted_indices:
            current_pts.append(polar_points[i])
            if i in gap_set:
                # 这里的 dist 是相对于芯片重心的，所以排序 Inner->Outer 绝对准确
                current_pts.sort(key=lambda p: p['dist']) 
                arms[arm_id] = current_pts
                current_pts = []
                arm_id += 1

        # --- Step 2: 锚点锁定 ---
        blank_arm_idx = -1
        for idx, pts in arms.items():
            cls_list = [p['cls'] for p in pts]
            if config.CLASS_ID_BLANK in cls_list:
                blank_arm_idx = idx
                break
        
        if blank_arm_idx == -1:
            print("[Warning] No Class 0 found. Fallback.")
            blank_arm_idx = 0

        # --- Step 3: 计算并应用刚性变换 ---
        M = self._get_manual_rigid_matrix(arms, blank_arm_idx, (chip_cx, chip_cy))
        final_img = cv2.warpAffine(img, M, (config.CANVAS_SIZE, config.CANVAS_SIZE), 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # --- Step 4: 变换坐标 & 切片 & 传递ClassID ---
        patches = []
        slice_points_real = [] # (x, y, cls)
        
        for i in range(4):
            # 顺序: N -> E -> S -> W
            real_idx = (blank_arm_idx + i) % 4
            pts = arms[real_idx] 
            
            raw_coords = np.array([p['pt'] for p in pts], dtype=np.float32).reshape(-1, 1, 2)
            transformed_coords = cv2.transform(raw_coords, M).reshape(-1, 2)
            
            # 同时遍历变换后的坐标和原始点(为了获取cls)
            for j, (tx, ty) in enumerate(transformed_coords):
                cls_id = pts[j]['cls']
                slice_points_real.append((tx, ty, cls_id)) # 关键：带上身份信息
                
                # 切片
                x1, y1 = int(tx - config.CROP_RADIUS), int(ty - config.CROP_RADIUS)
                x2, y2 = int(tx + config.CROP_RADIUS), int(ty + config.CROP_RADIUS)
                
                patch = final_img[max(0,y1):min(config.CANVAS_SIZE,y2), max(0,x1):min(config.CANVAS_SIZE,x2)]
                
                if patch.shape[0] != config.SLICE_SIZE[0] or patch.shape[1] != config.SLICE_SIZE[1]:
                    patch = cv2.resize(patch, config.SLICE_SIZE)
                    
                patches.append(patch)

        # --- Step 5: Debug 可视化 ---
        debug_vis = self._draw_debug(final_img, slice_points_real)

        return np.array(patches), debug_vis

    def _draw_debug(self, img, real_points):
        vis = img.copy()
        
        # 1. 蓝色实心点 (实际位置) + 红圈 (Class 0)
        for k, (rx, ry, r_cls) in enumerate(real_points):
            
            # 修正：只要是 Class 0，就画红圈！(不再依赖位置假设)
            if r_cls == config.CLASS_ID_BLANK:
                cv2.circle(vis, (int(rx), int(ry)), 25, (0, 0, 255), 3)
                cv2.putText(vis, "BLANK", (int(rx)-20, int(ry)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            cv2.circle(vis, (int(rx), int(ry)), 3, (255, 0, 0), -1) # 蓝点
            
            r = config.CROP_RADIUS
            cv2.rectangle(vis, (int(rx-r), int(ry-r)), (int(rx+r), int(ry+r)), (255, 255, 0), 1)

        return vis