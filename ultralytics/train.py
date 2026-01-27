from ultralytics import YOLO
import torch # 导入torch以检查GPU

def main():
    # --- 1. 检查设备 ---
    # 自动检测是否有可用的NVIDIA GPU (CUDA)，否则回退到CPU
    # (对应您在`how_to_install_yolo.md`中的疑问)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- 正在使用设备: {device} ---")
    
    # --- 2. 加载模型 ---
    # model=yolo11n.pt
    # 加载预训练的 yolo11n (nano) 模型。
    # 如果本地没有 yolo11n.pt, ultralytics 会自动下载它。
    model = YOLO("/home/asus515/PycharmProjects/YOLO_v11/ultralytics/yolo11n.pt")

    # --- 3. 运行训练 ---
    # 这里我们将您在命令行中的所有参数（`yolo_dataset_plan.md`）
    # 作为.train()方法的关键字参数(arguments)传入
    print("--- 开始训练 ---")
    results = model.train(
        # data=data.yaml
        data="/home/asus515/PycharmProjects/YOLO_v11/dataset/yolo_dataset/data.yaml",     # (!!!) 确保这个 .yaml 文件的路径是正确的
        
        # epochs=300
        epochs=300,           # (!!!) 为小数据集（`yolo_dataset_plan.md`）增加轮次
        
        # imgsz=1280
        imgsz=1280,           # (!!!) 使用高分辨率对抗小目标（`yolo_dataset_plan.md`）
        
        # batch=8
        batch=8,              # 较小的批量大小, 配合imgsz=1280防止显存溢出
        
        # augment=True
        augment=True,         # (!!!) 开启数据增强, 对抗小数据集（`yolo_dataset_plan.md`）过拟合
        
        # device=0 (或 'cuda', 或 'cpu')
        device=device,        # 自动选择最佳设备
        
        # (可选) 为您的训练任务命名
        name="chip_detection_run1" 
    )
    
    print("--- 训练完成 ---")
    print(f"训练结果保存在: {results.save_dir}")

if __name__ == '__main__':
    # 确保脚本被直接运行时才执行
    main()