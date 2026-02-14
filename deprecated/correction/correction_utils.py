import numpy as np
import cv2

# --- 步骤 2: 局部背景减除 (LBS) ---

def get_local_background(image, box, ring_width=5):
    """
    在包围盒(box)紧邻的“光环”区域内采样，计算局部背景RGB值。
    这用于消除芯片制造差异和局部阴影。

    Args:
        image (np.ndarray): 完整的(BGR)图像。
        box (list): [x1, y1, x2, y2] 格式的包围盒。
        ring_width (int): 向外采样的光环宽度（像素）。

    Returns:
        tuple: (R_bg, G_bg, B_bg) 局部背景的平均RGB值。
    """
    # 确保坐标是整数
    x1, y1, x2, y2 = map(int, box)
    
    # 定义内部盒 (腔室本身)
    inner_box = (x1, y1, x2, y2)
    
    # 定义外部盒 (向外扩大 ring_width)
    # np.clip确保坐标不会超出图像边界
    h, w = image.shape[:2]
    outer_x1 = np.clip(x1 - ring_width, 0, w - 1)
    outer_y1 = np.clip(y1 - ring_width, 0, h - 1)
    outer_x2 = np.clip(x2 + ring_width, 0, w - 1)
    outer_y2 = np.clip(y2 + ring_width, 0, h - 1)
    outer_box = (outer_x1, outer_y1, outer_x2, outer_y2)

    # 创建一个蒙版，只选中“光环”区域
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 填充外部盒为白色 (255)
    cv2.rectangle(mask, (outer_box[0], outer_box[1]), (outer_box[2], outer_box[3]), 255, -1)
    
    # 在蒙版上挖掉内部盒 (填充为黑色 0)
    cv2.rectangle(mask, (inner_box[0], inner_box[1]), (inner_box[2], inner_box[3]), 0, -1)
    
    # 计算蒙版区域的平均RGB值
    # cv2.mean会返回 (B_mean, G_mean, R_mean, Alpha_mean)
    mean_bgr = cv2.mean(image, mask=mask)
    
    # 我们只需要BGR (注意cv2的顺序是BGR)
    # 返回 (R, G, B) 顺序
    return (mean_bgr[2], mean_bgr[1], mean_bgr[0])

def calculate_pure_signal(image, box, ring_width=5):
    """
    步骤 2 (LBS) 的执行函数。
    计算 信号(S) - 背景(B)，得到“纯信号”。
    
    Args:
        image (np.ndarray): 完整的(BGR)图像。
        box (list): [x1, y1, x2, y2] 格式的包围盒。
        ring_width (int): 光环宽度。

    Returns:
        np.ndarray: [R_pure, G_pure, B_pure] 纯信号RGB向量。
    """
    x1, y1, x2, y2 = map(int, box)
    
    # 1. 提取腔室内部信号 (S)
    chamber_roi = image[y1:y2, x1:x2]
    if chamber_roi.size == 0:
        return np.array([0.0, 0.0, 0.0]) # 如果框无效，返回0
        
    mean_signal_bgr = cv2.mean(chamber_roi)
    signal_rgb = np.array([mean_signal_bgr[2], mean_signal_bgr[1], mean_signal_bgr[0]])
    
    # 2. 提取局部背景 (B)
    background_rgb = np.array(get_local_background(image, box, ring_width))
    
    # 3. 计算 纯信号 (S - B)
    pure_signal = signal_rgb - background_rgb
    
    # 裁剪到[0, 255]范围
    return np.clip(pure_signal, 0, 255)


# --- 步骤 3: 线性解算 (m 和 c) ---

def solve_linear_correction(ideal_anchor_dark, ideal_anchor_lit, observed_anchor_dark, observed_anchor_lit):
    """
    步骤 3 的执行函数。
    基于两个锚点，解算 y = mx + c 中的 m 和 c。
    这是针对 R, G, B 三个通道独立完成的。
    
    Args:
        ideal_anchor_dark (np.ndarray): [R,G,B] 理想暗室的“纯信号”
        ideal_anchor_lit (np.ndarray): [R,G,B] 理想亮室的“纯信号”
        observed_anchor_dark (np.ndarray): [R,G,B] 观测暗室的“纯信号”
        observed_anchor_lit (np.ndarray): [R,G,B] 观测亮室的“纯信号”

    Returns:
        tuple: (m_vector, c_vector), 每个都是[R,G,B]向量。
    """
    m_vector = np.ones(3)  # 乘性干扰 [m_R, m_G, m_B]
    c_vector = np.zeros(3) # 加性干扰 [c_R, c_G, c_B]
    
    epsilon = 1e-6 # 防止除以零

    for i in range(3): # 遍历 R, G, B
        ideal_dark = ideal_anchor_dark[i]
        ideal_lit = ideal_anchor_lit[i]
        obs_dark = observed_anchor_dark[i]
        obs_lit = observed_anchor_lit[i]

        delta_ideal = ideal_lit - ideal_dark
        delta_obs = obs_lit - obs_dark

        # 1. 解算 m (增益)
        if abs(delta_ideal) < epsilon:
            # 理想的暗点和亮点信号相同，无法解算增益，假设为1
            m = 1.0
        else:
            m = delta_obs / delta_ideal
        
        # 2. 解算 c (偏移)
        # c = obs_dark - m * ideal_dark
        # (使用两个点的平均值来计算c，更鲁棒)
        c = ( (obs_dark - m * ideal_dark) + (obs_lit - m * ideal_lit) ) / 2.0
        
        m_vector[i] = m
        c_vector[i] = c
        
    return m_vector, c_vector

# --- 步骤 4: 反向求解 ---

def apply_inverse_correction(observed_reaction_signal, m, c):
    """
    步骤 4 的执行函数。
    使用解算出的 m 和 c，反向求解“理想RGB”。
    Ideal = (Observed - c) / m
    
    Args:
        observed_reaction_signal (np.ndarray): [R,G,B] 反应室的“纯信号”
        m (np.ndarray): [m_R, m_G, m_B] 解算出的增益
        c (np.ndarray): [c_R, c_G, c_B] 解算出的偏移

    Returns:
        np.ndarray: [R_ideal, G_ideal, B_ideal] 预测的理想RGB值。
    """
    epsilon = 1e-6 # 再次防止除以零
    
    # Ideal = (Observed - c) / m
    predicted_ideal_rgb = (observed_reaction_signal - c) / (m + epsilon)
    
    # 裁剪，因为理想值也应该在0-255范围内
    return np.clip(predicted_ideal_rgb, 0, 255)