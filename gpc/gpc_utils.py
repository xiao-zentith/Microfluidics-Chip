import numpy as np
import math
from sklearn.cluster import KMeans

"""
GPC (几何后分类器) 的核心算法库
"""

# --- 1. 辅助函数 ---

def get_box_center(box):
    """辅助函数：计算YOLO包围盒(x1, y1, x2, y2)的中心点"""
    x1, y1, x2, y2 = box
    return np.array([ (x1 + x2) / 2, (y1 + y2) / 2 ])

def calculate_angle_degrees(vec1, vec2):
    """辅助函数：计算两个向量之间的相对角度（-180到180度）"""
    angle1 = math.atan2(vec1[1], vec1[0])
    angle2 = math.atan2(vec2[1], vec2[0])
    angle = angle1 - angle2
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle <= -math.pi:
        angle += 2 * math.pi
    return math.degrees(angle)

def assign_functions_to_arms(control_arm_vector, arm_vectors_map):
    """
    核心逻辑：根据您芯片的“刚性布局”先验知识，分配功能。
    
    (!!!) 注意：您必须根据您的真实芯片设计来修改这个函数。
    我在这里“假设”了一个布局，例如：
    - Glucose_Arm 在 Control 的 顺时针 90 度 (-90 deg)
    - Lipid_Arm 在 Control 的 180 度
    - Uric_Acid_Arm 在 Control 的 逆时针 90 度 (+90 deg)
    """
    
    # (这是您的“先验知识”配置 - `thesis_image_rag_plan.md`)
    PRIOR_KNOWLEDGE = {
        "Glucose_Arm": -90.0,   # 顺时针90度
        "Lipid_Arm": 180.0,
        "Uric_Acid_Arm": 90.0    # 逆时针90度
    }
    
    arm_functions = {}
    
    for arm_name_key, arm_vector in arm_vectors_map.items():
        relative_angle = calculate_angle_degrees(arm_vector, control_arm_vector)
        min_diff = float('inf')
        assigned_function = None
        
        for func_name, target_angle in PRIOR_KNOWLEDGE.items():
            diff = abs(relative_angle - target_angle)
            diff_wrapped = abs(360 - diff) # 环状差异
            final_diff = min(diff, diff_wrapped)
            
            if final_diff < min_diff:
                min_diff = final_diff
                assigned_function = func_name
        
        if assigned_function in arm_functions.values():
            print(f"--- 警告: GPC分配冲突！ {assigned_function} 被重复分配。")
            continue
            
        arm_functions[arm_name_key] = assigned_function
        
    return arm_functions

# --- 2. GPC 主函数 (可重用) ---

def run_gpc_classifier(yolo_boxes):
    """
    GPC (几何后分类器) 的主函数
    
    参数:
    - yolo_boxes: YOLOv11 (`train.py`) 的 boxes 对象
    
    返回:
    - a dictionary (如果成功), e.g.:
      {
        "Control_Blank": (box, center_point),
        "Control_Liquid": [ (box1, center1), (box2, center2) ],
        "Glucose_Arm": [ (boxA, cA), (boxB, cB), (boxC, cC) ],
        ...
      }
    - None (如果失败)
    """
    
    lit_chambers = []
    dark_chambers = []

    for box in yolo_boxes:
        class_id = int(box.cls[0])
        box_coords = box.xyxy[0].cpu().numpy()
        center_point = get_box_center(box_coords)
        chamber_data = (box_coords, center_point)

        if class_id == 1: # 0 = chamber_dark
            lit_chambers.append(chamber_data)
        elif class_id == 0: # 1 = chamber_lit
            dark_chambers.append(chamber_data)

    if len(dark_chambers) != 1:
        print(f"*** GPC错误: 致命失败! YOLO未能准确找到1个 'chamber_dark' (锚点)。")
        print(f"找到了 {len(dark_chambers)} 个。跳过。")
        return None
    
    if len(lit_chambers) != 11:
        print(f"*** GPC错误: 致命失败! YOLO未能准确找到11个 'chamber_lit'。")
        print(f"找到了 {len(lit_chambers)} 个。跳过。")
        return None
    
    anchor_blank = dark_chambers[0]
    center_dark = anchor_blank[1]
    
    lit_chambers_with_dist = []
    for lit_chamber in lit_chambers:
        center_lit = lit_chamber[1]
        distance = np.linalg.norm(center_lit - center_dark)
        lit_chambers_with_dist.append((lit_chamber, distance))
        
    lit_chambers_with_dist.sort(key=lambda x: x[1])
    
    anchor_liquid_1 = lit_chambers_with_dist[0][0]
    anchor_liquid_2 = lit_chambers_with_dist[1][0]
    reaction_chambers = [item[0] for item in lit_chambers_with_dist[2:]]
    
    if len(reaction_chambers) != 9:
        print(f"*** GPC错误: 腔室数量不匹配! 预期9个反应室, 实际 {len(reaction_chambers)} 个。")
        return None

    reaction_centers = np.array([chamber[1] for chamber in reaction_chambers])
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(reaction_centers)
    labels = kmeans.labels_
    
    all_centers = [center_dark, anchor_liquid_1[1], anchor_liquid_2[1]] + list(reaction_centers)
    total_centroid = np.mean(all_centers, axis=0)
    
    control_arm_centers = [center_dark, anchor_liquid_1[1], anchor_liquid_2[1]]
    control_arm_centroid = np.mean(control_arm_centers, axis=0)
    V_Control = control_arm_centroid - total_centroid

    arm_vectors_map = {}
    final_reaction_arms = {}
    
    for i in range(3):
        cluster_points = [reaction_chambers[j][1] for j, chamber in enumerate(reaction_chambers) if labels[j] == i]
        cluster_chambers = [chamber for j, chamber in enumerate(reaction_chambers) if labels[j] == i]
        
        arm_centroid = np.mean(cluster_points, axis=0)
        V_Arm = arm_centroid - total_centroid
        
        arm_key = f"Arm_{i}"
        arm_vectors_map[arm_key] = V_Arm
        final_reaction_arms[arm_key] = cluster_chambers

    arm_functions = assign_functions_to_arms(V_Control, arm_vectors_map)

    final_output = {
        "Control_Blank": anchor_blank,
        "Control_Liquid": [anchor_liquid_1, anchor_liquid_2]
    }
    
    for arm_key, function_name in arm_functions.items():
        final_output[function_name] = final_reaction_arms[arm_key]
        
    return final_output