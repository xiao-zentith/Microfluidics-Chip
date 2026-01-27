import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# (!!!) 导入我们新创建的 GPC 算法库
import gpc_utils 

# --- 1. 配置您的路径 ---
# (!!!) 关键：指向您刚刚训练好的模型权重
MODEL_PATH = Path("/home/asus515/PycharmProjects/YOLO_v11/ultralytics/runs/detect/chip_detection_run12/weights/best.pt")
# (!!!) 指向您用于“最终测试”的图像文件夹
TEST_IMAGES_DIR = Path("/home/asus515/PycharmProjects/YOLO_v11/dataset/correction_data/val/images_dirty/val")
# (!!!) 您希望保存“GPC 5分类”图像的输出文件夹
OUTPUT_DIR = Path("/home/asus515/PycharmProjects/YOLO_v11/visualize_result/dataset2/gpc")

# --- 2. 推理参数 ---
IMG_SIZE = 1280
CONF_THRESHOLD = 0.5 # YOLO的置信度阈值

# --- 3. 可视化主函数 ---
def run_inference_and_gpc_visualization():
    """
    主函数：加载模型，遍历测试图像，
    运行YOLO（`how_to_install_yolo.md`）获取2分类，
    调用GPC（`gpc_utils.py`）获取5分类，
    并保存可视化结果。
    """
    # 检查文件夹
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"*** 错误: 找不到模型文件! {MODEL_PATH} (来自 'train.py')")
        print("    请检查 'runs/detect/' 下的最新运行文件夹路径。")
        return
    if not TEST_IMAGES_DIR.exists():
        print(f"*** 错误: 找不到测试图像文件夹! {TEST_IMAGES_DIR} (来自 'split_dataset.py')")
        return

    # 加载模型
    print(f"正在加载模型: {MODEL_PATH}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL_PATH)
    model.to(device)
    print(f"模型加载完毕, 正在使用设备: {device}")

    # 获取所有测试图像
    image_files = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
    print(f"\n在 {len(image_files)} 张测试图像上运行GPC可视化...")

    # (!!!) GPC的可视化颜色和字体
    COLOR_MAP = {
        "Control_Blank": (255, 255, 255), # 白色
        "Control_Liquid": (200, 200, 200),# 灰色
        "Glucose_Arm": (0, 0, 255),    # 红色
        "Lipid_Arm": (0, 255, 0),      # 绿色
        "Uric_Acid_Arm": (255, 0, 0),    # 蓝色
        "Unknown_Arm": (0, 255, 255)  # 黄色 (用于错误)
    }
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    processed_count = 0
    failed_count = 0

    for img_path in image_files:
        print(f"  > 正在处理: {img_path.name}")
        
        # 步骤 1: 运行YOLOv11 (`how_to_install_yolo.md`) 推理
        # 获取2分类结果 (`chamber_lit` / `chamber_dark`)
        try:
            results = model.predict(
                source=str(img_path),
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                save=False, # 我们要自己绘制
                verbose=False # 关闭YOLO的啰嗦日志
            )
            yolo_result = results[0]
            original_image = yolo_result.orig_img # 获取原始OpenCV图像
        except Exception as e:
            print(f"    ! YOLO 推理失败: {e}")
            failed_count += 1
            continue
        
        # 步骤 2: (!!!) 调用外部的GPC算法 (`gpc_utils.py`)
        # (我们只传递YOLO的包围盒结果)
        try:
            gpc_results = gpc_utils.run_gpc_classifier(yolo_result.boxes)
        except Exception as e:
            print(f"    ! GPC 算法失败: {e}")
            gpc_results = None # 标记为失败
            
        
        if not gpc_results:
            # 如果GPC失败 (例如没找到1个暗室或11个亮室)
            failed_count += 1
            print(f"    ! GPC 分类失败 (约束条件不满足，例如腔室数量错误)")
            output_path = OUTPUT_DIR / f"{img_path.stem}_GPC_FAILED.jpg"
            cv2.imwrite(str(output_path), original_image)
            continue

        # 步骤 3: (!!!) 绘制GPC的5分类结果 (您的核心需求)
        annotated_image_gpc = original_image.copy()

        for function_name, chambers in gpc_results.items():
            # 为未知臂分配颜色
            if "Unknown_Arm" in function_name:
                color = COLOR_MAP["Unknown_Arm"]
            else:
                color = COLOR_MAP.get(function_name, COLOR_MAP["Unknown_Arm"])
            
            # 将 'chambers' 统一为列表
            # `chambers` 的结构是 [(box, center), (box, center), ...]
            # 或者是 `Control_Blank` 的 (box, center)
            if isinstance(chambers, list):
                chamber_list = chambers
            else: # (这是 Control_Blank)
                chamber_list = [chambers] 
            
            for chamber_data in chamber_list:
                box, center = chamber_data
                x1, y1, x2, y2 = map(int, box)
                
                # 绘制包围盒
                cv2.rectangle(annotated_image_gpc, (x1, y1), (x2, y2), color, FONT_THICKNESS)
                
                # 绘制功能标签 (e.g., "Glucose_Arm")
                cv2.putText(
                    annotated_image_gpc, 
                    function_name, 
                    (x1, y1 - 10), # 放在框的上方
                    FONT, 
                    FONT_SCALE, 
                    color, 
                    FONT_THICKNESS
                )

        # 步骤 4: 保存GPC可视化图像
        output_filename = f"{img_path.stem}_GPC_annotated.jpg"
        output_path = OUTPUT_DIR / output_filename
        cv2.imwrite(str(output_path), annotated_image_gpc)
        processed_count += 1

    print("\n--- GPC可视化完成! ---")
    print(f"成功处理并保存: {processed_count} 张图像")
    print(f"失败/跳过: {failed_count} 张图像")
    print(f"所有GPC 5分类标注的图像已保存到: {OUTPUT_DIR}")

# --- 4. 运行主函数 ---
if __name__ == '__main__':
    run_inference_and_gpc_visualization()