import os
import cv2
import numpy as np
from ultralytics import YOLO
import gpc_utils         # (!!!) 依赖 gpc_utils.py
import correction_utils  # (!!!) 依赖 correction_utils.py
from pathlib import Path
import re

# --- 1. 配置 (您需要修改的部分) ---

# (!!!) 替换为您训练好的 YOLOv11 模型的 .pt 文件路径
MODEL_PATH = Path("/home/asus515/PycharmProjects/YOLO_v11/ultralytics/runs/detect/chip_detection_run12/weights/best.pt")

# (!!!) 替换为您的“成对”数据集(thesis_image_rag_plan.md)的路径
DATASET_DIR = Path("/home/asus515/PycharmProjects/YOLO_v11/dataset/correction_data/val") # 假设的父文件夹
DIRTY_IMAGE_DIR = DATASET_DIR / "images_dirty" / "val"  # "问题" (X)
CLEAN_IMAGE_DIR = DATASET_DIR / "images_clean" / "val"  # "答案" (Y)

# (!!!) 命名约定 (用于查找 S001_... -> S001_clean.jpg)
# (这是一个正则表达式, 匹配 "S001_dirty_...")
NAMING_REGEX = re.compile(r"^(S\d+)_dirty_.*") 
CLEAN_SUFFIX = "_clean.jpg" # (假设所有净图都是.jpg)

# --- (可选) GPC 和 校正 的配置 ---
REACTION_CHAMBERS_TO_TEST = ["Glucose_Arm", "Lipid_Arm", "Uric_Acid_Arm"]
ANCHOR_DARK_ID = "Control_Blank"
ANCHOR_LIT_IDS = ["Control_Liquid"]
RING_WIDTH = 5

# --- 2. 辅助函数: 提取“纯信号” ---
def get_all_pure_signals(image, yolo_model, gpc_results):
    """
    辅助函数: 从 GPC (gpc_utils.py) 结果中提取所有腔室的“纯信号”。
    """
    signals = {}
    
    # 提取所有臂 (包括反应臂和基线臂的亮室)
    all_arms_keys = REACTION_CHAMBERS_TO_TEST + ANCHOR_LIT_IDS
    for arm_name in all_arms_keys:
        if arm_name in gpc_results:
            arm_signals = []
            # GPC (gpc_utils.py) 返回 [(box, center), ...]
            for (box, _) in gpc_results[arm_name]:
                pure_signal = correction_utils.calculate_pure_signal(image, box, RING_WIDTH)
                arm_signals.append(pure_signal)
            
            # 取臂上所有腔室的平均值
            if arm_signals:
                signals[arm_name] = np.mean(arm_signals, axis=0)
    
    # 提取唯一的暗锚点
    if ANCHOR_DARK_ID in gpc_results:
        # GPC (gpc_utils.py) 返回 (box, center)
        box_dark, _ = gpc_results[ANCHOR_DARK_ID] 
        signals[ANCHOR_DARK_ID] = correction_utils.calculate_pure_signal(image, box_dark, RING_WIDTH)
        
    return signals

# --- 3. 主测试流程 ---
def main():
    print("开始“白盒”光学校正 (v2) 流水线测试...")
    
    # 1. 加载模型
    print(f"加载YOLOv11模型 (how_to_install_yolo.md) 从: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"*** 错误: 找不到模型文件 {MODEL_PATH}。请运行 train.py。")
        return
    yolo_model = YOLO(MODEL_PATH)
    
    # 2. 遍历所有“脏图”(dirty_dataset_collection_plan.md) 进行测试
    print(f"\n开始遍历“脏图”进行测试 (来自 {DIRTY_IMAGE_DIR})...")
    
    all_mses = [] # 存储所有腔室的MSE
    
    image_files = list(DIRTY_IMAGE_DIR.glob("*.jpg")) + list(DIRTY_IMAGE_DIR.glob("*.png"))
    if not image_files:
        print(f"*** 错误: 在 {DIRTY_IMAGE_DIR} 中找不到任何“脏图”。")
        return

    for dirty_path in image_files:
        dirty_name = dirty_path.name
        print(f"\n--- 正在处理: {dirty_name} ---")
        
        # 3.1 (!!!) 查找对应的“净图”
        match = NAMING_REGEX.match(dirty_name)
        if not match:
            print(f"  [警告] 脏图 {dirty_name} 命名不规范 (不匹配 'S001_dirty_...'), 跳过。")
            continue
            
        base_id = match.group(1) # (例如 "S001")
        clean_name = base_id + CLEAN_SUFFIX
        clean_path = CLEAN_IMAGE_DIR / clean_name
        
        # 3.2 加载“成对”图像 (thesis_image_rag_plan.md)
        img_dirty = cv2.imread(str(dirty_path))
        img_clean = cv2.imread(str(clean_path))
        
        if img_dirty is None:
            print(f"  [警告] 无法加载脏图，跳过 {dirty_name}")
            continue
        if img_clean is None:
            print(f"  [警告] 找不到对应的“净图” {clean_path}，跳过")
            continue
            
        # 3.3 在两张图上分别运行 GPC (gpc_utils.py)
        try:
            results_dirty = yolo_model(img_dirty, verbose=False)
            gpc_results_dirty = gpc_utils.run_gpc_classifier(results_dirty[0].boxes)
            
            results_clean = yolo_model(img_clean, verbose=False)
            gpc_results_clean = gpc_utils.run_gpc_classifier(results_clean[0].boxes)
        except Exception as e:
            print(f"  [警告] GPC在 {dirty_name} 或 {clean_name} 上失败: {e}")
            continue

        if gpc_results_dirty is None or gpc_results_clean is None:
            print(f"  [警告] GPC在 {dirty_name} 或 {clean_name} 上未返回有效结果，跳过。")
            continue

        # 3.4 提取“纯信号” (correction_utils.py)
        try:
            # (A) 从“净图”提取“理想”信号
            gt_signals = get_all_pure_signals(img_clean, yolo_model, gpc_results_clean)
            
            # (B) 从“脏图”提取“观测”信号
            obs_signals = get_all_pure_signals(img_dirty, yolo_model, gpc_results_dirty)

            # 确保所有锚点都已找到
            ideal_dark_baseline = gt_signals[ANCHOR_DARK_ID]
            ideal_lit_baseline = gt_signals[ANCHOR_LIT_IDS[0]] # GPC (gpc_utils.py) 里是列表, 我们取平均后的
            obs_dark_baseline = obs_signals[ANCHOR_DARK_ID]
            obs_lit_baseline = obs_signals[ANCHOR_LIT_IDS[0]]
            
        except KeyError as e:
            print(f"  [警告] 无法在 {dirty_name} 或 {clean_name} 中提取到锚点: {e}")
            continue
        except Exception as e:
            print(f"  [警告] 提取纯信号时出错: {e}")
            continue
            
        # 3.5 步骤 3 (解算) - 为这张“脏图”单独解算 m 和 c
        m, c = correction_utils.solve_linear_correction(
            ideal_dark_baseline, 
            ideal_lit_baseline, 
            obs_dark_baseline, 
            obs_lit_baseline
        )
            
        # 3.6 步骤 4 (反向求解) + 步骤 5 (对比)
        for arm_name in REACTION_CHAMBERS_TO_TEST:
            if arm_name not in obs_signals or arm_name not in gt_signals:
                print(f"  [警告] {arm_name} 在脏图或净图中未找到")
                continue
                
            # (A) 获取观测值 (来自脏图)
            obs_reaction_signal = obs_signals[arm_name]
            
            # (B) 反向求解 "理想RGB"
            pred_ideal_rgb = correction_utils.apply_inverse_correction(obs_reaction_signal, m, c)
            
            # (C) 获取金标准 "理想RGB" (来自净图)
            gt_ideal_rgb = gt_signals[arm_name]
            
            # (D) 计算误差
            mse = np.mean((pred_ideal_rgb - gt_ideal_rgb) ** 2)
            all_mses.append(mse)
            
            # 打印对比结果
            print(f"  {arm_name}:")
            print(f"    - 预测 理想RGB: {np.round(pred_ideal_rgb, 2)}")
            print(f"    - 真实 理想RGB: {np.round(gt_ideal_rgb, 2)}")
            print(f"    - 均方误差(MSE): {mse:.2f}")

    # 4. 最终报告
    if all_mses:
        final_rmse = np.sqrt(np.mean(all_mses))
        print("\n" + "="*30)
        print(f"“白盒”校正方案（thesis_image_rag_plan.md 6.0）测试完成。")
        print(f"总均方根误差 (RMSE) (RGB通道): {final_rmse:.4f}")
        print("="*30)
    else:
        print("\n[!!!] 测试失败，没有一个腔室被成功处理。")


if __name__ == "__main__":
    # 确保 GPC 库 (gpc_utils.py) 已正确导入
    if not hasattr(gpc_utils, 'run_gpc_classifier'):
        print("错误: 找不到 gpc_utils.py 或其 'run_gpc_classifier' 函数。")
        print("请确保 gpc_utils.py 与此脚本在同一文件夹中。")
    elif not hasattr(correction_utils, 'calculate_pure_signal'):
        print("错误: 找不到 correction_utils.py 或其 'calculate_pure_signal' 函数。")
        print("请确保 correction_utils.py 与此脚本在同一文件夹中。")
    else:
        main()