# detector.py

from ultralytics import YOLO
import numpy as np
import config

class ChamberDetector:
    def __init__(self):
        print(f"[Detector] Loading YOLO model from {config.YOLO_WEIGHTS_PATH}...")
        self.model = YOLO(config.YOLO_WEIGHTS_PATH)

    def detect(self, img):
        """
        输入图像，返回检测到的中心点和类别
        :return: (points, classes) 
                 points: np.array shape (N, 2)
                 classes: np.array shape (N,)
        """
        # 运行推理 (conf 设低一点防止漏检，后续靠逻辑筛选)
        results = self.model.predict(img, conf=0.5, verbose=False)
        result = results[0]

        if len(result.boxes) == 0:
            return np.array([]), np.array([])

        # 获取 xywh (中心点 x, y, 宽, 高)
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        points = boxes_xywh[:, :2] # 只取 center_x, center_y

        # 获取类别
        classes = result.boxes.cls.cpu().numpy().astype(int)

        return points, classes