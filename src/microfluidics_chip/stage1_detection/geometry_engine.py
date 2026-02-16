"""
十字几何校正引擎
继承自 v1.0 的 preprocess/utils.py::CrossGeometryEngine
修改遵循 v1.1 P0 强制规范：
- process() 必须返回四元组 (aligned_image, chamber_slices, transform_params, debug_vis)

核心算法逻辑（继承自 v1.0）：
1. 真实坐标跟随（Real-Coordinate Following）：切片精准居中
2. 身份跟随（Identity Following）：红圈基于 Class ID 绘制
3. 重心排序（Centroid Sorting）：基于芯片重心排序，解决排序倒置问题
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Any
from ..core.types import ChamberDetection, TransformParams
from ..core.config import GeometryConfig
from ..core.logger import get_logger

logger = get_logger("stage1_detection.geometry_engine")


class CrossGeometryEngine:
    """
    十字几何校正引擎（V10.0 完善版）
    
    P0 强制接口：
    - process() 返回四元组 (aligned_image, chamber_slices, transform_params, debug_vis)
    """
    
    def __init__(self, config: GeometryConfig):
        """
        初始化几何引擎
        
        :param config: 几何配置对象（依赖注入）
        """
        self.config = config
        self.center = (config.canvas_size // 2, config.canvas_size // 2)
        self.ideal_slice_centers = self._generate_ideal_centers()
        self.last_process_debug: Dict[str, Any] = {}
        logger.info(f"GeometryEngine initialized (canvas={config.canvas_size})")

    @staticmethod
    def _as_homo(points: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        return np.hstack([points.astype(np.float32), ones])

    @staticmethod
    def _affine_apply(M: np.ndarray, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points.astype(np.float32)
        homo = CrossGeometryEngine._as_homo(points)
        return (M.astype(np.float32) @ homo.T).T

    def _crop_patch(self, image: np.ndarray, cx: float, cy: float) -> np.ndarray:
        x1 = int(round(cx - self.config.crop_radius))
        y1 = int(round(cy - self.config.crop_radius))
        x2 = int(round(cx + self.config.crop_radius))
        y2 = int(round(cy + self.config.crop_radius))

        patch = image[
            max(0, y1):min(image.shape[0], y2),
            max(0, x1):min(image.shape[1], x2)
        ]
        if patch.size == 0:
            return np.zeros((self.config.slice_size[1], self.config.slice_size[0], 3), dtype=np.uint8)
        if patch.shape[0] != self.config.slice_size[0] or patch.shape[1] != self.config.slice_size[1]:
            patch = cv2.resize(patch, self.config.slice_size)
        return patch

    def _compute_canonical_points(self, template_points: np.ndarray) -> np.ndarray:
        pts = np.asarray(template_points, dtype=np.float32)
        centroid = np.mean(pts, axis=0, keepdims=True)
        centered = pts - centroid
        canvas_center = np.array([[float(self.center[0]), float(self.center[1])]], dtype=np.float32)
        return centered + canvas_center
    
    def _generate_ideal_centers(self) -> List[Tuple[int, int]]:
        """生成理想十字布局的12个腔室中心点"""
        centers = []
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        
        for dx, dy in directions:
            for i in range(3):
                dist = self.config.ideal_center_gap + i * self.config.ideal_chamber_step
                cx = self.center[0] + dx * dist
                cy = self.center[1] + dy * dist
                centers.append((cx, cy))
        
        return centers
    
    def _get_manual_rigid_matrix(
        self,
        arms: Dict[int, List[Dict]],
        blank_arm_idx: int,
        src_centroid: Tuple[float, float]
    ) -> np.ndarray:
        """
        计算刚性变换矩阵（T -> R -> S -> T'）
        
        :param arms: 四臂数据（字典，key=arm_id, value=点列表）
        :param blank_arm_idx: 空白臂索引
        :param src_centroid: 芯片重心 (cx, cy)
        :return: 2x3 变换矩阵
        """
        src_cx, src_cy = src_centroid
        
        # Source Angle（基于 Blank Arm 方向）
        blank_pts = [p['pt'] for p in arms[blank_arm_idx]]
        b_avg_x = np.mean([p[0] for p in blank_pts])
        b_avg_y = np.mean([p[1] for p in blank_pts])
        src_angle_rad = math.atan2(b_avg_y - src_cy, b_avg_x - src_cx)
        
        # Source Scale（基于到芯片重心的距离）
        all_pts = []
        for pts in arms.values():
            for p in pts:
                all_pts.append(p['pt'])
        dists = [math.sqrt((p[0]-src_cx)**2 + (p[1]-src_cy)**2) for p in all_pts]
        src_scale = np.mean(dists)
        
        # Target Params
        dst_cx, dst_cy = self.center
        dst_angle_rad = math.radians(-90)  # 指向上方（Top）
        dst_scale = np.mean([
            self.config.ideal_center_gap,
            self.config.ideal_center_gap + self.config.ideal_chamber_step,
            self.config.ideal_center_gap + 2 * self.config.ideal_chamber_step
        ])
        
        # Delta
        delta_angle = dst_angle_rad - src_angle_rad
        scale_factor = dst_scale / src_scale
        
        # 矩阵构造：T1 -> R -> S -> T2
        T1 = np.array([[1, 0, -src_cx], [0, 1, -src_cy], [0, 0, 1]])
        c, s = math.cos(delta_angle), math.sin(delta_angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        S = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
        T2 = np.array([[1, 0, dst_cx], [0, 1, dst_cy], [0, 0, 1]])
        
        M_3x3 = T2 @ S @ R @ T1
        return M_3x3[:2, :]  # 返回 2x3 仿射矩阵
    
    def process(
        self,
        img: np.ndarray,
        detections: List[ChamberDetection],
        canonical_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[TransformParams], Optional[np.ndarray]]:
        """
        处理图像：拓扑排序 -> 刚性变换 -> 切片
        
        P0 强制接口：返回四元组
        
        :param img: 输入图像 (H, W, 3) uint8
        :param detections: ChamberDetection 列表（P0 强制类型）
        :return: (aligned_image, chamber_slices, transform_params, debug_vis)
                 - aligned_image: np.ndarray (canvas_size, canvas_size, 3)
                 - chamber_slices: np.ndarray (12, H, W, 3)
                 - transform_params: TransformParams
                 - debug_vis: np.ndarray（可视化图像）
        """
        # 新路径：模板驱动 canonical 对齐（不再依赖十字理想参数）
        if canonical_context is not None:
            return self._process_template_driven(
                img=img,
                detections=detections,
                canonical_context=canonical_context
            )

        # 兼容旧路径：十字假设几何处理
        self.last_process_debug = {"mode": "legacy_cross"}

        # 校验：至少需要 12 个检测
        if len(detections) < 12:
            logger.warning(f"Insufficient detections: {len(detections)} < 12")
            return None, None, None, None
        
        # 提取关键点和类别（从 ChamberDetection 列表）
        keypoints = [det.center for det in detections]
        classes = [det.class_id for det in detections]
        
        h, w = img.shape[:2]
        
        # ==================== Step 0: 计算芯片重心 ====================
        kp_np = np.array(keypoints)
        chip_cx = np.mean(kp_np[:, 0])
        chip_cy = np.mean(kp_np[:, 1])
        
        # ==================== Step 1: 拓扑排序 ====================
        polar_points = []
        for i, (x, y) in enumerate(keypoints):
            # 基于芯片重心计算极坐标
            dx, dy = x - chip_cx, y - chip_cy
            angle = math.degrees(math.atan2(dy, dx)) % 360
            dist = math.sqrt(dx**2 + dy**2)
            polar_points.append({
                'angle': angle,
                'dist': dist,
                'cls': classes[i],
                'pt': (x, y)
            })
        
        polar_points.sort(key=lambda p: p['angle'])
        
        # 找到 4 个最大角度间隙（旋臂分界线）
        angles = [p['angle'] for p in polar_points]
        gaps = []
        for i in range(len(angles)):
            diff = (angles[(i + 1) % len(angles)] - angles[i]) % 360
            gaps.append((diff, i))
        
        gaps.sort(key=lambda x: x[0], reverse=True)
        top_4_indices = sorted([g[1] for g in gaps[:4]])
        
        # 按角度顺序分组为 4 个旋臂
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
                # 按距离排序（Inner -> Outer）
                current_pts.sort(key=lambda p: p['dist'])
                arms[arm_id] = current_pts
                current_pts = []
                arm_id += 1
        
        # ==================== Step 2: 锚点锁定 ====================
        blank_arm_idx = -1
        for idx, pts in arms.items():
            cls_list = [p['cls'] for p in pts]
            if self.config.class_id_blank in cls_list:  # 使用配置中的 class_id
                blank_arm_idx = idx
                break
        
        if blank_arm_idx == -1:
            logger.warning("No blank chamber found. Using fallback arm 0")
            blank_arm_idx = 0
        
        # ==================== Step 3: 计算并应用刚性变换 ====================
        M = self._get_manual_rigid_matrix(arms, blank_arm_idx, (chip_cx, chip_cy))
        aligned_image = cv2.warpAffine(
            img, M,
            (self.config.canvas_size, self.config.canvas_size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # ==================== Step 4: 变换坐标 & 切片 ====================
        patches = []
        slice_points_real = []  # (x, y, cls)
        
        for i in range(4):
            # 顺序：N -> E -> S -> W
            real_idx = (blank_arm_idx + i) % 4
            pts = arms[real_idx]
            
            # 变换坐标
            raw_coords = np.array([p['pt'] for p in pts], dtype=np.float32).reshape(-1, 1, 2)
            transformed_coords = cv2.transform(raw_coords, M).reshape(-1, 2)
            
            # 提取切片（保持身份信息）
            for j, (tx, ty) in enumerate(transformed_coords):
                cls_id = pts[j]['cls']
                slice_points_real.append((tx, ty, cls_id))
                
                # 切片裁剪
                x1, y1 = int(tx - self.config.crop_radius), int(ty - self.config.crop_radius)
                x2, y2 = int(tx + self.config.crop_radius), int(ty + self.config.crop_radius)
                
                patch = aligned_image[
                    max(0, y1):min(self.config.canvas_size, y2),
                    max(0, x1):min(self.config.canvas_size, x2)
                ]
                
                # 尺寸校正
                if patch.size == 0:
                    logger.warning(f"Empty patch for chamber {i} (tx={tx:.1f}, ty={ty:.1f}). Filling black.")
                    patch = np.zeros((self.config.slice_size[1], self.config.slice_size[0], 3), dtype=np.uint8)
                elif patch.shape[0] != self.config.slice_size[0] or patch.shape[1] != self.config.slice_size[1]:
                    patch = cv2.resize(patch, self.config.slice_size)
                
                patches.append(patch)
        
        chamber_slices = np.array(patches)
        
        # ==================== Step 5: Debug 可视化 ====================
        debug_vis = self._draw_debug(aligned_image, slice_points_real)
        
        # ==================== P0: 构造 TransformParams ====================
        # 提取旋转角和缩放系数（从矩阵反推）
        rotation_angle = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        scale_factor = math.sqrt(M[0, 0]**2 + M[1, 0]**2)
        
        transform_params = TransformParams(
            rotation_angle=rotation_angle,
            scale_factor=scale_factor,
            chip_centroid=(chip_cx, chip_cy),
            blank_arm_index=blank_arm_idx,
            matrix=M.tolist()  # 可选：保存完整矩阵
        )

        self.last_process_debug = {
            "mode": "legacy_cross",
            "blank_arm_index": int(blank_arm_idx),
            "rotation_angle": float(rotation_angle),
            "scale_factor": float(scale_factor),
            "slice_centers_aligned": [
                [float(x), float(y)] for (x, y, _) in slice_points_real
            ],
        }
        
        logger.debug(f"Processed: rotation={rotation_angle:.2f}°, scale={scale_factor:.3f}")
        
        # P0 强制返回：四元组
        return aligned_image, chamber_slices, transform_params, debug_vis

    def _process_template_driven(
        self,
        img: np.ndarray,
        detections: List[ChamberDetection],
        canonical_context: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[TransformParams], Optional[np.ndarray]]:
        """
        模板驱动 canonical 几何路径：
        - 输入 raw 拟合点 + 模板 canonical 点
        - 求 raw->canonical 变换
        - warp 后按 canonical 点切片（严格12点）
        """
        if len(detections) < 12:
            logger.warning(f"Insufficient detections for template-driven mode: {len(detections)} < 12")
            return None, None, None, None

        template_points = np.asarray(canonical_context.get("template_points"), dtype=np.float32)
        topology_fitted_raw = np.asarray(canonical_context.get("fitted_points_raw"), dtype=np.float32)
        final_raw = np.asarray(
            canonical_context.get("final_points_raw", canonical_context.get("fitted_points_raw")),
            dtype=np.float32
        )
        if template_points.ndim != 2 or template_points.shape[0] < 12 or template_points.shape[1] != 2:
            logger.warning("Invalid template_points for template-driven geometry")
            return None, None, None, None
        if final_raw.ndim != 2 or final_raw.shape[0] < 12 or final_raw.shape[1] != 2:
            logger.warning("Invalid final_points_raw for template-driven geometry")
            return None, None, None, None

        template_points = template_points[:12]
        final_raw = final_raw[:12]
        if topology_fitted_raw.ndim == 2 and topology_fitted_raw.shape[0] >= 12 and topology_fitted_raw.shape[1] == 2:
            topology_fitted_raw = topology_fitted_raw[:12]
        else:
            topology_fitted_raw = final_raw.copy()
        canonical_pts = self._compute_canonical_points(template_points)

        # 优先 similarity，必要时 affine
        M, inlier_mask = cv2.estimateAffinePartial2D(
            final_raw.reshape(-1, 1, 2),
            canonical_pts.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        transform_type = "similarity"
        if M is None:
            M, inlier_mask = cv2.estimateAffine2D(
                final_raw.reshape(-1, 1, 2),
                canonical_pts.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.99,
            )
            transform_type = "affine"
        if M is None:
            logger.warning("Template-driven affine solve failed")
            return None, None, None, None

        aligned = cv2.warpAffine(
            img,
            M,
            (self.config.canvas_size, self.config.canvas_size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # canonical slices (主输出)
        canonical_slices: List[np.ndarray] = []
        for (cx, cy) in canonical_pts:
            canonical_slices.append(self._crop_patch(aligned, float(cx), float(cy)))
        chamber_slices = np.asarray(canonical_slices)

        # raw slices (调试输出)
        raw_slices: List[np.ndarray] = []
        for (rx, ry) in final_raw:
            raw_slices.append(self._crop_patch(img, float(rx), float(ry)))
        raw_slices_np = np.asarray(raw_slices)

        # 变换一致性与投影误差
        invM = cv2.invertAffineTransform(M.astype(np.float32))
        projected_raw = self._affine_apply(invM, canonical_pts)
        reproj_vec = np.linalg.norm(projected_raw - final_raw, axis=1)
        reproj_mean = float(np.mean(reproj_vec))
        reproj_max = float(np.max(reproj_vec))

        projected_canonical = self._affine_apply(M.astype(np.float32), final_raw)
        center_offset_vec = np.linalg.norm(projected_canonical - canonical_pts, axis=1)
        center_offset_mean = float(np.mean(center_offset_vec))
        center_offset_max = float(np.max(center_offset_vec))

        # debug overlay (raw)
        debug_overlay = img.copy()
        r = int(self.config.crop_radius)
        for i in range(12):
            rx, ry = final_raw[i]
            px, py = projected_raw[i]
            cv2.circle(debug_overlay, (int(round(rx)), int(round(ry))), 4, (255, 80, 0), -1, cv2.LINE_AA)
            cv2.circle(debug_overlay, (int(round(px)), int(round(py))), 7, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(
                debug_overlay,
                (int(round(rx)) - r, int(round(ry)) - r),
                (int(round(rx)) + r, int(round(ry)) + r),
                (255, 0, 255),
                1,
            )
            cv2.putText(
                debug_overlay,
                str(i),
                (int(round(rx)) + 6, int(round(ry)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 180),
                1,
                cv2.LINE_AA,
            )
            # 若 topology 拟合点与最终点不一致，叠加拟合点供审计
            tx, ty = topology_fitted_raw[i]
            if abs(float(tx) - float(rx)) > 1e-3 or abs(float(ty) - float(ry)) > 1e-3:
                cv2.circle(debug_overlay, (int(round(tx)), int(round(ty))), 3, (255, 0, 0), -1, cv2.LINE_AA)

        blank_idx = canonical_context.get("blank_idx")
        if blank_idx is not None and 0 <= int(blank_idx) < 12:
            bx, by = final_raw[int(blank_idx)]
            cv2.circle(debug_overlay, (int(round(bx)), int(round(by))), 20, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_overlay, "BLANK", (int(round(bx)) - 24, int(round(by)) - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        # aligned debug（沿用原可视化风格）
        debug_vis = aligned.copy()
        for i, (cx, cy) in enumerate(canonical_pts):
            cv2.circle(debug_vis, (int(round(cx)), int(round(cy))), 3, (255, 0, 0), -1)
            cv2.rectangle(
                debug_vis,
                (int(round(cx)) - r, int(round(cy)) - r),
                (int(round(cx)) + r, int(round(cy)) + r),
                (255, 255, 0),
                1,
            )
            cv2.putText(
                debug_vis,
                str(i),
                (int(round(cx)) + 5, int(round(cy)) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 180),
                1,
                cv2.LINE_AA,
            )

        rotation_angle = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        scale_factor = math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        blank_arm = int(blank_idx) // 3 if blank_idx is not None else 0
        chip_centroid = tuple(np.mean(final_raw, axis=0).tolist())

        transform_params = TransformParams(
            rotation_angle=float(rotation_angle),
            scale_factor=float(scale_factor),
            chip_centroid=(float(chip_centroid[0]), float(chip_centroid[1])),
            blank_arm_index=int(blank_arm),
            matrix=M.tolist(),
        )

        inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
        template_ids = canonical_context.get("template_ids") or [str(i) for i in range(12)]
        semantic_order = canonical_context.get("semantic_order_clockwise_from_blank") or []

        self.last_process_debug = {
            "mode": "template_canonical",
            "transform_type": transform_type,
            "transform_matrix_raw_to_canonical": M.tolist(),
            "inverse_transform_matrix_canonical_to_raw": invM.tolist(),
            "n_inliers_transform": inliers,
            "fitted_points_raw": final_raw.tolist(),
            "topology_fitted_points_raw": topology_fitted_raw.tolist(),
            "canonical_points": canonical_pts.tolist(),
            "projected_raw_points": projected_raw.tolist(),
            "template_ids": template_ids,
            "semantic_order_clockwise_from_blank": semantic_order,
            "point_source": canonical_context.get("point_source"),
            "det_fit_distance_px": canonical_context.get("det_fit_distance_px"),
            "semantic_roles_by_template": canonical_context.get("semantic_roles_by_template"),
            "semantic_arm_role_by_template": canonical_context.get("semantic_arm_role_by_template"),
            "blank_idx": int(blank_idx) if blank_idx is not None else None,
            "reprojection_error_mean_px": reproj_mean,
            "reprojection_error_max_px": reproj_max,
            "slice_center_offset_mean_px": center_offset_mean,
            "slice_center_offset_max_px": center_offset_max,
            "raw_slices": raw_slices_np,
            "debug_overlay_raw": debug_overlay,
        }

        logger.debug(
            "Template-driven geometry: inliers=%d reproj_mean=%.3f reproj_max=%.3f center_off_max=%.3f",
            inliers, reproj_mean, reproj_max, center_offset_max
        )
        return aligned, chamber_slices, transform_params, debug_vis
    
    def _draw_debug(self, img: np.ndarray, real_points: List[Tuple]) -> np.ndarray:
        """
        绘制调试可视化
        
        :param img: 对齐后的图像
        :param real_points: [(x, y, cls_id), ...]
        :return: 可视化图像
        """
        vis = img.copy()
        
        for k, (rx, ry, r_cls) in enumerate(real_points):
            # 标记空白腔（红圈）
            if r_cls == self.config.class_id_blank:
                cv2.circle(vis, (int(rx), int(ry)), 25, (0, 0, 255), 3)
                cv2.putText(
                    vis, "BLANK",
                    (int(rx)-20, int(ry)-30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )
            
            # 蓝点标记中心
            cv2.circle(vis, (int(rx), int(ry)), 3, (255, 0, 0), -1)
            
            # 黄框标记切片区域
            r = self.config.crop_radius
            cv2.rectangle(
                vis,
                (int(rx-r), int(ry-r)),
                (int(rx+r), int(ry+r)),
                (255, 255, 0), 1
            )
        
        return vis
