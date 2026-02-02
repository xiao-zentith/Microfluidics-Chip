# YOLO 检测精度提升指南

> **目标**: 确保 YOLO 在所有图像上稳定检测到 12 个腔室，mAP@0.5 > 0.95

---

## 🎯 当前问题诊断

### 检测YOLO性能

```bash
# 方法1: 使用quick_validation.py
python scripts/quick_validation.py dataset/chip001 -o runs/validation

# 方法2: 手动测试
python -c "
from microfluidics_chip.stage1_detection.detector import ChamberDetector
from microfluidics_chip.core.config import get_default_config
import cv2

config = get_default_config()
detector = ChamberDetector(config.stage1.yolo)

img = cv2.imread('test.png')
detections = detector.detect(img)
print(f'检测到 {len(detections)}/12 个腔室')
for i, det in enumerate(detections):
    print(f'  [{i}] class={det.class_id}, conf={det.confidence:.3f}')
"
```

**问题分类：**
| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 检测数 < 12 | 置信度阈值过高 | [方法1](#方法1降低置信度阈值) |
| 检测数 > 12 | 误检测/重复检测 | [方法2](#方法2nms优化) |
| 漏检特定腔室 | 训练数据不足 | [方法3](#方法3数据增强) |
| 光照变化敏感 | 缺少光照多样性 | [方法4](#方法4光照鲁棒性) |

---

## 方法1：降低置信度阈值

**适用场景**: 检测数少于12个，但漏检的腔室仍可见

### 修改配置文件

编辑 `configs/default.yaml`:

```yaml
stage1:
  yolo:
    model_path: "weights/yolo_chambers.pt"
    conf_threshold: 0.3   # 从0.5降低到0.3
    iou_threshold: 0.45   # NMS阈值
    max_det: 20           # 最大检测数（允许一些冗余）
```

### 动态测试最佳阈值

```python
import cv2
from microfluidics_chip.stage1_detection.detector import ChamberDetector
from microfluidics_chip.core.config import YOLOConfig

img = cv2.imread("test.png")

for conf in [0.2, 0.3, 0.4, 0.5, 0.6]:
    config = YOLOConfig(
        model_path="weights/yolo_chambers.pt",
        conf_threshold=conf
    )
    detector = ChamberDetector(config)
    dets = detector.detect(img)
    print(f"conf={conf}: {len(dets)} chambers")
```

**建议阈值：**
- 训练良好的模型：0.5
- 一般模型：0.3-0.4
- 调试阶段：0.2（会有误检，需配合NMS）

---

## 方法2：NMS 优化

**适用场景**: 同一个腔室被检测多次（bounding box重叠）

### 调整 IOU 阈值

```yaml
yolo:
  iou_threshold: 0.3   # 更激进的NMS（从0.45降低）
```

**原理**: IoU阈值越低，NMS越倾向于合并重叠框

### 后处理去重

如果仍有重复，可在 `detector.py` 中添加额外去重：

```python
def remove_duplicates_by_distance(detections, min_distance=30):
    """根据中心点距离去重"""
    filtered = []
    for det in detections:
        # 检查是否与已有检测重复
        is_duplicate = False
        for existing in filtered:
            dist = np.linalg.norm(
                np.array(det.center) - np.array(existing.center)
            )
            if dist < min_distance:
                # 保留置信度更高的
                if det.confidence > existing.confidence:
                    filtered.remove(existing)
                    filtered.append(det)
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(det)
    
    return filtered
```

---

## 方法3：数据增强

**适用场景**: 训练数据少，泛化能力不足

### 3.1 离线数据增强

创建增强脚本 `scripts/augment_yolo_data.py`:

```python
import albumentations as A
import cv2
from pathlib import Path

# 定义增强pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 对每张图像生成5个增强版本
for img_path in Path("data/stage1_detection/yolo_v1/images").glob("*.png"):
    img = cv2.imread(str(img_path))
    
    # 读取对应的标注
    label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    # ... (解析YOLO格式标注)
    
    for i in range(5):
        augmented = transform(image=img, bboxes=bboxes, class_labels=labels)
        # 保存增强图像和标注
```

### 3.2 在线数据增强（Ultralytics内置）

编辑YOLO训练配置 `yolo_train.yaml`:

```yaml
# 数据增强
hsv_h: 0.015  # 色调抖动
hsv_s: 0.7    # 饱和度抖动
hsv_v: 0.4    # 亮度抖动

degrees: 10   # 旋转角度
translate: 0.1  # 平移
scale: 0.5    # 缩放
shear: 0.0    # 剪切
perspective: 0.0  # 透视变换

flipud: 0.5   # 垂直翻转概率
fliplr: 0.5   # 水平翻转概率

mosaic: 1.0   # Mosaic增强（强烈推荐）
mixup: 0.1    # MixUp增强
```

---

## 方法4：光照鲁棒性训练

**适用场景**: 模型对光照变化敏感

### 4.1 收集多光照条件数据

```
data/stage1_detection/yolo_v1/
├── bright/     # 强光条件
├── normal/     # 正常光照
├── dark/       # 弱光条件
└── shadow/     # 局部阴影
```

### 4.2 合成光照变化数据

```python
import cv2
import numpy as np

def simulate_lighting_variations(img):
    """模拟光照变化"""
    variations = []
    
    # 1. 全局亮度调整
    for gamma in [0.6, 0.8, 1.2, 1.4]:
        adjusted = np.power(img / 255.0, gamma) * 255.0
        variations.append(adjusted.astype(np.uint8))
    
    # 2. 局部光照梯度
    h, w = img.shape[:2]
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    for angle in [0, 45, 90, 135]:
        rad = np.deg2rad(angle)
        gradient = 0.5 + 0.5 * (np.sin(rad) * X + np.cos(rad) * Y) / max(h, w)
        lit = (img * gradient[..., None]).clip(0, 255).astype(np.uint8)
        variations.append(lit)
    
    # 3. 局部阴影
    for _ in range(3):
        mask = np.ones((h, w), dtype=np.float32)
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(50, 150)
        cv2.circle(mask, (cx, cy), radius, 0.5, -1)
        shadowed = (img * mask[..., None]).clip(0, 255).astype(np.uint8)
        variations.append(shadowed)
    
    return variations
```

---

## 方法5：重新训练 YOLO（推荐）

**适用场景**: 数据充足（>100张标注图像）

### 5.1 准备数据集

```
data/stage1_detection/yolo_v1/
├── images/
│   ├── train/
│   │   ├── chip001.png
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── chip001.txt  # YOLO格式标注
    │   └── ...
    └── val/
        └── ...
```

**YOLO 标注格式** (`chip001.txt`):
```
0 0.5 0.5 0.1 0.1    # class_id center_x center_y width height (归一化)
0 0.3 0.4 0.1 0.1
...
```

### 5.2 训练脚本

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # nano版本，速度快

# 训练
results = model.train(
    data='yolo_data.yaml',    # 数据集配置
    epochs=100,
    imgsz=640,
    batch=16,
    device='0',               # GPU ID
    patience=20,              # Early stopping
    project='runs/yolo_train',
    name='chambers_v2',
    
    # 数据增强（重要！）
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    mosaic=1.0,
    mixup=0.1,
)

# 验证
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50}")

# 导出
model.export(format='onnx')  # 可选：导出为ONNX加速推理
```

**数据集配置** (`yolo_data.yaml`):
```yaml
path: data/stage1_detection/yolo_v1
train: images/train
val: images/val

nc: 1  # 类别数（只有chamber一个类）
names: ['chamber']
```

---

## 方法6：容错机制（推荐配合使用）

**适用场景**: YOLO精度提升有限，需要增强鲁棒性

### 6.1 检测后验证与修复

在 `geometry_engine.py` 中添加：

```python
def validate_and_repair_detections(detections, image_shape):
    """验证并修复检测结果"""
    h, w = image_shape[:2]
    
    # 1. 检查数量
    if len(detections) < 12:
        logger.warning(f"检测数不足: {len(detections)}/12")
        # 尝试使用模板匹配补全（基于十字几何）
        detections = try_complete_by_template(detections, image_shape)
    
    elif len(detections) > 12:
        logger.warning(f"检测数过多: {len(detections)}/12")
        # 根据置信度和几何约束筛选
        detections = filter_by_confidence_and_geometry(detections)
    
    # 2. 验证几何一致性
    if len(detections) == 12:
        # 检查是否符合十字几何（4个旋臂）
        if not check_cross_geometry(detections):
            logger.warning("几何结构异常")
            return None
    
    return detections


def check_cross_geometry(detections):
    """检查十字几何约束"""
    # 计算重心
    centers = np.array([det.center for det in detections])
    centroid = centers.mean(axis=0)
    
    # 计算极角
    angles = np.arctan2(
        centers[:, 1] - centroid[1],
        centers[:, 0] - centroid[0]
    )
    angles = np.rad2deg(angles) % 360
    angles_sorted = np.sort(angles)
    
    # 检查是否有4个聚类（每个旋臂3个腔室）
    gaps = np.diff(angles_sorted)
    large_gaps = gaps > 30  # 旋臂间隙应 > 30度
    
    return large_gaps.sum() == 4  # 应有4个大间隙
```

### 6.2 降级策略

```python
def process_with_fallback(image, detector, config):
    """带降级策略的处理"""
    
    # 尝试1: 标准检测
    detections = detector.detect(image)
    
    if len(detections) == 12:
        return process_normal(image, detections, config)
    
    # 尝试2: 降低阈值retry
    logger.warning("尝试降低置信度阈值")
    detector_low_conf = ChamberDetector(
        YOLOConfig(
            model_path=config.yolo.model_path,
            conf_threshold=0.2  # 降低阈值
        )
    )
    detections = detector_low_conf.detect(image)
    
    if len(detections) == 12:
        return process_normal(image, detections, config)
    
    # 尝试3: 跳过该图像
    logger.error(f"无法修复检测结果，跳过该图像")
    return None
```

---

## 📊 评估 YOLO 性能

### 计算 mAP

```python
from ultralytics import YOLO

model = YOLO('weights/yolo_chambers.pt')
metrics = model.val(data='yolo_data.yaml')

print(f"Precision: {metrics.box.p:.3f}")
print(f"Recall:    {metrics.box.r:.3f}")
print(f"mAP@0.5:   {metrics.box.map50:.3f}")  # 目标 > 0.95
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
```

### 可视化混淆矩阵

```python
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('weights/yolo_chambers.pt')
results = model.val(data='yolo_data.yaml', plots=True)

# 查看结果
# runs/val/confusion_matrix.png
# runs/val/P_curve.png  # Precision曲线
# runs/val/R_curve.png  # Recall曲线
```

---

## ✅ 推荐优化顺序

1. **快速验证** (5分钟)
   ```bash
   python scripts/quick_validation.py dataset/chip001
   ```

2. **调整阈值** (10分钟)
   - 尝试 `conf_threshold` 从 0.5 → 0.3
   - 观察检测数变化

3. **数据增强** (1小时)
   - 使用 Albumentations 生成 5x 数据

4. **重新训练** (2-4小时，如果数据充足)
   - 收集 >= 100 张标注图像
   - 使用 Mosaic + MixUp 增强

5. **添加容错** (30分钟)
   - 在 `geometry_engine.py` 添加验证逻辑

---

## 🎯 最终目标

| 指标 | 目标值 | 当前状态 |
|------|--------|---------|
| 检测成功率 | 100% | ? |
| mAP@0.5 | > 0.95 | ? |
| 单图推理时间 | < 50ms | ? |

**验证命令:**
```bash
python scripts/quick_validation.py dataset/chip001 --skip-yolo-check
```
