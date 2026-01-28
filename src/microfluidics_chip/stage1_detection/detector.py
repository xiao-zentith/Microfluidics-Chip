"""
YOLO 腔室检测器
继承自 v1.0 的 preprocess/detector.py
修改遵循 v1.1 P0 强制规范：
- detect() 必须返回 List[ChamberDetection]
"""

from ultralytics import YOLO
import numpy as np
from typing import List
from ..core.types import ChamberDetection
from ..core.config import YOLOConfig
from ..core.logger import get_logger

logger = get_logger("stage1_detection.detector")


class ChamberDetector:
    """
    YOLO 目标检测器
    
    P0 强制接口：
    - detect() 返回 List[ChamberDetection]（不再是 tuple）
    """
    
    def __init__(self, config: YOLOConfig):
        """
        初始化 YOLO 模型
        
        :param config: YOLO 配置对象（依赖注入）
        """
        self.config = config
        logger.info(f"Loading YOLO model from {config.weights_path}...")
        self.model = YOLO(config.weights_path)
        self.model.to(config.device)
        logger.info(f"YOLO model loaded on {config.device}")
    
    def detect(self, img: np.ndarray) -> List[ChamberDetection]:
        """
        检测图像中的腔室
        
        P0 强制接口：必须返回 List[ChamberDetection]
        
        :param img: 输入图像 (H, W, 3) uint8
        :return: ChamberDetection 列表（按检测顺序）
        
        示例：
        >>> detections = detector.detect(image)
        >>> for det in detections:
        >>>     print(f"Center: {det.center}, Class: {det.class_id}")
        """
        # 运行 YOLO 推理（conf 设低一点防止漏检）
        results = self.model.predict(
            img,
            conf=self.config.confidence_threshold,
            verbose=False
        )
        result = results[0]
        
        # 如果没有检测到任何目标，返回空列表
        if len(result.boxes) == 0:
            logger.warning("No chambers detected in image")
            return []
        
        # 提取检测结果
        boxes_xywh = result.boxes.xywh.cpu().numpy()  # (N, 4) [cx, cy, w, h]
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4) [x1, y1, x2, y2]
        classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
        confidences = result.boxes.conf.cpu().numpy()  # (N,)
        
        # 构造 ChamberDetection 列表（P0 规范）
        detections = []
        for i in range(len(boxes_xywh)):
            cx, cy, w, h = boxes_xywh[i]
            x1, y1, x2, y2 = boxes_xyxy[i]
            
            detection = ChamberDetection(
                bbox=(int(x1), int(y1), int(w), int(h)),  # (x, y, w, h)
                center=(float(cx), float(cy)),            # (cx, cy)
                class_id=int(classes[i]),                 # 0=blank, 1=lit
                confidence=float(confidences[i])          # [0, 1]
            )
            detections.append(detection)
        
        logger.debug(f"Detected {len(detections)} chambers")
        return detections
    
    def __repr__(self) -> str:
        return (f"ChamberDetector(weights='{self.config.weights_path}', "
                f"device='{self.config.device}', "
                f"conf_thresh={self.config.confidence_threshold})")
