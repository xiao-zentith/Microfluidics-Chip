# debug_sort.py
import cv2

from detector import ChamberDetector
from utils import CrossGeometryEngine

def main():
    # 替换你的图片路径
    img_path = "/home/asus515/PycharmProjects/YOLO_v11/preprocess_result/debug_vis/GT_chip3.jpg"
    
    detector = ChamberDetector()
    engine = CrossGeometryEngine()
    
    img = cv2.imread(img_path)
    points, classes = detector.detect(img)
    
    # 调用带日志的排序
    sorted_pts = engine.sort_and_anchor(points, classes, img)
    
    # 可视化排序结果
    vis = img.copy()
    # 理想顺序: 0,1,2 是 Top(Blank)
    for i, pt in enumerate(sorted_pts):
        cx, cy = int(pt[0]), int(pt[1])
        
        # 0-2 (Blank) 画红色
        if i < 3:
            color = (0, 0, 255)
            scale = 1.0
        else:
            color = (0, 255, 0)
            scale = 0.6
            
        cv2.circle(vis, (cx, cy), 10, color, 2)
        cv2.putText(vis, str(i), (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        
    cv2.imwrite("check_sort_order.jpg", vis)
    print("\n结果已保存至 check_sort_order.jpg")

if __name__ == "__main__":
    main()