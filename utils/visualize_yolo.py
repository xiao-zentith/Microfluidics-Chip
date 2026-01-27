import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# --- 1. 配置路径 ---
# (!!!) 关键：指向您刚刚训练好的模型权重
MODEL_PATH = Path("/home/asus515/PycharmProjects/YOLO_v11/ultralytics/runs/detect/chip_detection_run12/weights/best.pt")

# (!!!) 指向您用于“最终测试”的图像文件夹
TEST_IMAGES_DIR = Path("/home/asus515/PycharmProjects/YOLO_v11/dataset/correction_data/val/images_dirty/val")

# (!!!) 关键：自动推导标签文件夹的路径
# (基于 `yolo_dataset_plan.md` 的平行结构)
TEST_LABELS_DIR = TEST_IMAGES_DIR.parent.parent / "labels" / TEST_IMAGES_DIR.name

# (!!!) 新的输出文件夹，只保存YOLO 2分类的结果
OUTPUT_DIR = Path("/home/asus515/PycharmProjects/YOLO_v11/visualize_result/dataset2/yolo")

# --- 2. 推理参数 ---
IMG_SIZE = 1280
CONF_THRESHOLD = 0.5 # YOLO的置信度阈值

# --- 3. 可视化主函数 ---
def run_yolo_visualization():
    """
    主函数：加载标准(detect)模型，遍历测试图像。
    生成 _pred.jpg (YOLO的 4-coord 预测)
    和 _gt.jpg (Ground Truth 8-coord OBB 标签)
    
    (!!!) 已修正：_gt.jpg 现在也会显示正确的2分类颜色。
    """
    # 检查文件夹
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"*** 错误: 找不到模型文件! {MODEL_PATH}")
        print("    请确保您指向了 'runs/detect/' 下的正确路径。")
        return
    # if not TEST_IMAGES_DIR.exists() or not TEST_LABELS_DIR.exists():
    #     print(f"*** 错误: 找不到测试图像或标签文件夹! {TEST_LABELS_DIR}")
    #     return

    # 加载模型
    print(f"正在加载模型: {MODEL_PATH}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL_PATH) 
    model.to(device)
    print(f"模型加载完毕, 正在使用设备: {device}")
    
    CLASS_NAMES = model.names
    print(f"检测到模型类别: {CLASS_NAMES}")
    
    # (!!!) 关键：Pred 和 GT 共同使用这个颜色表
    COLOR_MAP = {
        CLASS_NAMES.get(0, "Class_0"): (0, 255, 255),  # 类别0 (lit) -> 黄色
        CLASS_NAMES.get(1, "Class_1"): (255, 0, 255), # 类别1 (dark) -> 紫色
    }
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.4 
    FONT_THICKNESS = 1

    processed_count = 0
    image_files = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
    print(f"\n在 {len(image_files)} 张测试图像上运行YOLO可视化...")

    for img_path in image_files:
        print(f"  > 正在处理: {img_path.name}")
        
        original_image = cv2.imread(str(img_path))
        if original_image is None: continue
        
        pred_image = original_image.copy()
        gt_image = original_image.copy()
        H, W, _ = original_image.shape

        # --- A. 生成 "Prediction" 图像 (标准 4-coord Box) ---
        # (这部分逻辑不变)
        try:
            results = model.predict(
                source=original_image,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                save=False,
                verbose=False
            )
            yolo_result = results[0]

            if yolo_result.boxes is None:
                print("    ! 警告: 模型没有返回 .boxes 结果。")
            else:
                for box in yolo_result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    class_name = CLASS_NAMES.get(cls_id, "UNKNOWN")
                    color = COLOR_MAP.get(class_name, (0, 0, 255))
                    label = f"{class_name}: {conf:.2f}"
                    
                    cv2.rectangle(pred_image, (x1, y1), (x2, y2), color, FONT_THICKNESS + 1)
                    text_pos = (x1, y1 - 10)
                    cv2.putText(pred_image, label, text_pos, FONT, FONT_SCALE, color, FONT_THICKNESS)
            
            pred_filename = f"{img_path.stem}_pred.jpg"
            cv2.imwrite(str(OUTPUT_DIR / pred_filename), pred_image)
            
        except Exception as e:
            print(f"    ! YOLO 推理失败: {e}")
            continue

        # --- B. 生成 "Ground Truth" (Labels) 图像 (OBB 8-coord) ---
        # (!!!) 这部分逻辑已修正 (!!!)
        label_path = TEST_LABELS_DIR / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"    ! 警告: 找不到对应的标签文件 {label_path.name}")
        else:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    try:
                        parts = line.strip().split()
                        cls_id = int(parts[0])
                        
                        # (!!!) 修正点 1: 根据 cls_id 获取正确的颜色
                        class_name = CLASS_NAMES.get(cls_id, "UNKNOWN")
                        color = COLOR_MAP.get(class_name, (0, 0, 255)) # 使用与Pred相同的颜色
                        
                        if len(parts[1:]) != 8:
                            print(f"    ! 警告: 标签行格式错误 (非8个坐标): {line}")
                            continue
                            
                        coords_norm = list(map(float, parts[1:]))
                        points_norm = np.array(coords_norm).reshape(4, 2)
                        points_px = (points_norm * np.array([W, H])).astype(int)
                        
                        label = f"{class_name} (GT)"
                        
                        # (!!!) 修正点 2: 使用获取到的 'color'
                        cv2.polylines(gt_image, [points_px], isClosed=True, color=color, thickness=FONT_THICKNESS + 1)
                        text_pos = (points_px[0][0], points_px[0][1] - 10)
                        
                        # (!!!) 修正点 3: 使用获取到的 'color'
                        cv2.putText(gt_image, label, text_pos, FONT, FONT_SCALE, color, FONT_THICKNESS)
                    
                    except Exception as e:
                        print(f"    ! 警告: 解析标签行失败: '{line}'. 错误: {e}")

            gt_filename = f"{img_path.stem}_gt.jpg"
            cv2.imwrite(str(OUTPUT_DIR / gt_filename), gt_image)

        processed_count += 1

    print(f"\n--- YOLO (Detect) vs OBB (GT) 可视化完成! ---")
    print(f"成功处理并保存: {processed_count} 张图像 (每张图 x2 = _pred.jpg + _gt.jpg)")
    print(f"所有YOLO 2分类标注的图像已保存到: {OUTPUT_DIR}")

# --- 4. 运行主函数 ---
if __name__ == '__main__':
    run_yolo_visualization()