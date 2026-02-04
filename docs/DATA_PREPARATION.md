# 数据集准备与训练指南

> **目标**: 将原始拍摄图像转换为训练数据，并训练 Stage2 光照校正模型

---

## 📁 第一步：准备数据集目录

### 要求的目录结构

```
data/
├── stage1_detection/            # Stage1: YOLO 目标检测数据（独立管理）
│   └── yolo_v1/                 # YOLO 数据集版本 1
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── data.yaml
│
├── stage2_correction/           # Stage2: UNet 光照校正数据（独立管理）
│   ├── microfluidics_v1/        # ✅ 数据集名称（推荐以实验版本命名）
│   │   ├── raw/                 # 原始拍摄数据
│   │   │   ├── chip001/
│   │   │   │   ├── gt.png
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── processed/           # 预处理后的NPZ文件
│   │       ├── training.npz
│   │       └── test.npz
│   └── microfluidics_v2/        # 示例：未来可添加更多数据集
│       └── ...
│
└── experiments/                 # 训练输出
    ├── 2024-01-30_baseline/
    └── 2024-01-31_augmented/
```

### 命名规则

| 文件类型 | 支持的命名模式 | 说明 |
|---------|---------------|------|
| **GT图像** | `gt.png`, `gt.jpg`, `GT.png` | 每个芯片目录**必须有1张** |
| **Dirty图像** | `dirty_*.png`, `dirty_*.jpg`, `noisy_*.png` | 每个芯片可有**多张** |

### 拍摄建议

1. **GT图像**：在均匀照明下拍摄，避免阴影和反光
2. **Dirty图像**：模拟真实使用场景，可以：
   - 调整光源角度
   - 改变环境光照
   - 添加局部阴影
   - 每个芯片建议至少5张dirty图像

---

## 🛠️ (可选) 辅助工具：一键重命名

如果你拍摄的照片文件名杂乱（例如 `IMG_2023.jpg`, `DSC_001.jpg`），可以使用 `scripts/rename_dataset.py` 脚本一键标准化命名。

### 功能
- 自动识别 GT 图像（根据文件名关键词或文件大小）
- 自动将其余图像重命名为 `dirty_01.jpg`, `dirty_02.jpg`...
- 自动备份原始文件名

### 用法

```bash
# 1. 预览重命名计划（DRY-RUN，不执行）
python scripts/rename_dataset.py dataset/chip001 --dry-run

# 2. 执行重命名
python scripts/rename_dataset.py dataset/chip001

# 3. 如果自动识别GT错误，手动指定
python scripts/rename_dataset.py dataset/chip001 --gt-image IMG_9999.jpg
```

---

## � 第二步：Stage1 YOLO 数据集准备与训练

### YOLO 数据集标注格式

YOLO 使用 **YOLO 格式标注**（`.txt` 文件），每行一个检测框：

```
<class_id> <center_x> <center_y> <width> <height>
```

**坐标归一化**：所有值都在 [0, 1] 范围内，相对于图像尺寸。

**示例** (`chip001.txt`)：
```
0 0.342 0.512 0.085 0.092   # 类别0: chamber_dark
1 0.658 0.488 0.081 0.089   # 类别1: chamber_lit
...
```

### 数据集组织

```
data/stage1_detection/yolo_v1/
├── images/
│   ├── train/                 # 训练图像
│   │   ├── chip001.png
│   │   ├── chip002.png
│   │   └── ...
│   └── val/                   # 验证图像（可选，可用 train 代替）
│       └── ...
├── labels/
│   ├── train/                 # 训练标注
│   │   ├── chip001.txt        # 与图像同名
│   │   ├── chip002.txt
│   │   └── ...
│   └── val/
│       └── ...
└── data.yaml                  # 数据集配置文件
```

### 配置文件 `data.yaml`

```yaml
train: images/train
val: images/train   # 如果没有单独验证集，可以用训练集

nc: 2  # 类别数量
names: 
  0: chamber_dark   # 类别0: 暗腔室
  1: chamber_lit    # 类别1: 亮腔室
```

> **💡 提示**：如果你没有时间标注验证集，直接让 `val: images/train`。训练时会在训练集上做验证，虽然不够严格，但可以看到拟合效果。

### 🚀 (强烈推荐) 使用离线增强扩充数据集

为了解决光照和距离带来的域偏移（Domain Shift），建议先运行该脚本对数据集进行 5 倍扩充。这能利用 Stage2 的物理光照模型 (ISP) 让 YOLO 见过各种极端光照。

**功能**: 生成不同光照、白平衡、噪声的 "Dirty" 图像，保留原始标签。

```bash
python scripts/augment_yolo_dataset.py \
    --input data/stage1_detection/yolo_v1/images/train \
    --multiplier 5
```

**效果**:
- 训练集数量：**N -> 5N** (例：300 -> 1500 张)
- 覆盖率：大幅提升对阴影、过曝、低光照的检测能力。

---

### YOLO 训练命令

#### 方法 1：使用 Ultralytics CLI（推荐）

```bash
# 新建训练（从预训练模型开始）
yolo detect train \
    data=data/stage1_detection/yolo_v1/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=runs/yolo_train \
    name=chambers_v1 \
    # --- 数据增强参数 (默认已开启，此处显式设置示范) ---
    hsv_h=0.015    # 色调 (Hue) 增强
    hsv_s=0.7      # 饱和度 (Saturation) 增强
    hsv_v=0.4      # 亮度 (Value) 增强
    degrees=10.0   # 旋转 (+/- 10度)
    translate=0.1  # 平移 (+/- 0.1)
    scale=0.5      # 缩放 (+/- 0.5)
    flipud=0.0     # 垂直翻转概率 (显微镜图像推荐设为 0.5)
    fliplr=0.5     # 水平翻转概率
    mosaic=1.0     # Mosaic 增强 (拼接4张图，极强，推荐开启)
    mixup=0.0      # MixUp 增强 (混合2张图，推荐关闭或设低)
```

#### 方法 2：Python 脚本

创建 `scripts/train_yolo.py`：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # nano 版本，快速

# 训练
results = model.train(
    data='data/stage1_detection/yolo_v1/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project='runs/yolo_train',
    name='chambers_v1',
    
    # 数据增强（推荐）
    hsv_h=0.015,      # 色调抖动
    hsv_s=0.7,        # 饱和度
    hsv_v=0.4,        # 亮度
    degrees=10,       # 旋转
    mosaic=1.0,       # Mosaic 增强
    mixup=0.1,        # MixUp 增强
)

print(f"训练完成，mAP@0.5: {results.box.map50}")
```

运行：
```bash
python scripts/train_yolo.py
```

### 训练输出

```
runs/yolo_train/chambers_v1/
├── weights/
│   ├── best.pt                 # 最佳模型（按 mAP）
│   └── last.pt                 # 最终模型
├── results.png                 # 训练曲线
├── confusion_matrix.png        # 混淆矩阵
├── val_batch0_labels.jpg       # 验证集真值
├── val_batch0_pred.jpg         # 验证集预测（肉眼看效果）
└── args.yaml                   # 训练参数记录
```

> **👀 肉眼可视化**：查看 `val_batch0_pred.jpg` 查看模型在验证集上的预测效果！

### YOLO 模型验证

```bash
# 在验证集上评估
yolo detect val \
    model=runs/yolo_train/chambers_v1/weights/best.pt \
    data=data/stage1_detection/yolo_v1/data.yaml

# 单张图像推理
yolo detect predict \
    model=runs/yolo_train/chambers_v1/weights/best.pt \
    source=test_image.png \
    conf=0.5
```

### 将训练好的模型部署到项目

训练完成后，将最佳模型复制到项目权重目录：

```bash
# Windows
copy runs\yolo_train\chambers_v1\weights\best.pt weights\yolo\best.pt

# Linux/Mac
cp runs/yolo_train/chambers_v1/weights/best.pt weights/yolo/best.pt
```

然后更新 `configs/default.yaml`：
```yaml
stage1:
  yolo:
    weights_path: "weights/yolo/best.pt"
```

---

## 🔧 第三步：Stage2 UNet 数据准备

### 基础用法

```bash
python scripts/prepare_training_data.py data/stage2_correction/microfluidics_v1/raw -o data/stage2_correction/microfluidics_v1/processed/training.npz
```

### 使用离线增强 (v1.2)

```bash
# 5倍ISP增强 (推荐)
python scripts/prepare_training_data.py data/stage2_correction/microfluidics_v1/raw -o data/stage2_correction/microfluidics_v1/processed/training.npz \
    --augment --aug-multiplier 5
```

**增强内容**:
- 光照场 (渐晕 + 方向性光源)
- 白平衡漂移
- 曝光变化
- Gamma校正
- Shot Noise (光子计数模型)

### 完整参数

```bash
python scripts/prepare_training_data.py \
    data/stage2_correction/microfluidics_v1/raw \
    --output data/stage2_correction/microfluidics_v1/processed/training.npz \
    --config configs/default.yaml \
    --augment \
    --aug-multiplier 5 \
    --no-debug
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset_dir` | 数据集根目录（位置参数） | - |
| `-o, --output` | 输出NPZ文件路径 | `processed_data/training.npz` |
| `-c, --config` | 配置文件 | `None`（使用默认配置） |
| `--augment` | 启用离线ISP增强 (v1.2) | 禁用 |
| `--aug-multiplier` | 增强倍数 (1-10) | 5 |
| `--no-debug` | 不保存调试图像 | 默认保存 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `data/stage2_correction/microfluidics_v1/processed/training.npz` | 训练数据（target_in, ref_in, labels） |
| `chip*/debug_gt.png` | GT图像的检测+几何校正可视化（调试用） |
| `chip*/debug_dirty_*.png` | Dirty图像的可视化（调试用） |

### NPZ 数据格式

| Key | 形状 | 取值范围 | 说明 |
|-----|------|---------|------|
| `target_in` | (N, H, W, 3) | [0, 1] | 待校正图像（dirty切片） |
| `ref_in` | (N, H, W, 3) | [0, 1] | 参考图像（GT基准腔室平均） |
| `labels` | (N, H, W, 3) | [0, 1] | 真值图像（GT切片） |

**验证数据格式：**
```bash
python scripts/verify_npz_format.py processed_data/training.npz
```

---

## 🎯 第四步：训练 Stage2 UNet 模型

### 日常训练（推荐）

```bash
python scripts/train_stage2.py processed_data/training.npz -o runs/my_training -e 100
```

**参数说明：**
```bash
python scripts/train_stage2.py data/stage2_correction/microfluidics_v1/processed/training.npz \
    --output runs/my_training \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001 \
    --device cuda \
    --roi-radius 20 \
    --edge-weight 0.1 \
    --lambda-cos 0.2
```

### 消融实验训练

```bash
# 双流模型（Our Method）
python scripts/train_experiments.py -c configs/experiments/ablation_a_dual.yaml

# 单流模型（Baseline）
python scripts/train_experiments.py -c configs/experiments/ablation_a_single.yaml
```

### 训练输出

```
runs/my_training/
├── weights/
│   ├── best_model.pth          # 最佳模型（按PSNR）
│   ├── final_model.pth         # 最终模型
│   └── checkpoint_epoch*.pth   # 定期检查点
├── visualizations/
│   └── epoch_*.png             # 训练过程可视化
├── training_curves.png         # Loss/PSNR曲线
└── training.log                # 训练日志
```

---

## 📊 第五步：评估 Stage2 模型

### 准备测试集

```bash
python scripts/prepare_training_data.py dataset/test -o processed_data/test.npz
```

### 评估单个模型

```bash
python scripts/evaluate_experiments.py \
    -e runs/my_training \
    -t processed_data/test.npz \
    -o results/evaluation.json
```

### 对比多个模型

```bash
python scripts/evaluate_experiments.py \
    -e runs/exp_dual runs/exp_single \
    -t processed_data/test.npz \
    -o results/comparison.json
```

**输出文件：**
| 文件 | 说明 |
|------|------|
| `evaluation.json` | JSON格式指标 |
| `evaluation.md` | Markdown表格 |
| `evaluation_comparison.png` | 对比柱状图 |
| `evaluation_roi_comparison.png` | ROI vs Edge RMSE |

---

## 🚀 第六步：使用训练好的模型

### CLI 推理

```bash
# Stage1 + Stage2 完整流程
python -m microfluidics_chip.pipelines.cli stage1 input.png -o output/

# 仅 Stage2 校正
python -m microfluidics_chip.pipelines.cli stage2 \
    input.png \
    runs/my_training/weights/best_model.pth \
    -o output/
```

### Python API

```python
from pathlib import Path
from microfluidics_chip.stage2_correction.models import RefGuidedUNet
import torch
import cv2

# 加载模型
checkpoint = torch.load("runs/my_training/weights/best_model.pth")
model = RefGuidedUNet()
model.load_state_dict(checkpoint['model'])
model.eval()

# 推理
signal = cv2.imread("dirty_chamber.png")  # (H, W, 3)
reference = cv2.imread("gt_chamber.png")

# ... (需要转换为tensor并预处理)
```

---

## ⚠️ 常见问题

### 1. YOLO检测失败

**错误**: `Insufficient GT detections: 8 < 12`

**原因**: YOLO模型未找到12个腔室

**解决**:
1. 检查YOLO模型路径：`configs/default.yaml` 中 `yolo.model_path`
2. 降低置信度：`yolo.conf_threshold: 0.3`
3. 检查图像质量：确保12个腔室清晰可见

### 2. Reference 形状错误（已修复）

**错误**: `RuntimeError: The size of tensor a (3) must match...`

**原因**: 之前的bug，reference_combined形状为(N_ref, H, W, 3)

**状态**: ✅ 已在本次修复中解决

### 3. 数据量不足

**建议**:
- 每个芯片至少5张dirty图像
- 至少10个不同的芯片
- 总样本量建议 > 500

或使用数据增强：
```bash
python scripts/prepare_mixed_dataset.py  # 合成数据增强
```

---

## 📝 配置说明

### `configs/default.yaml`

```yaml
stage2:
  reference_chambers: [0, 1, 2]    # 基准腔室索引（前3个）
  reference_mode: "average"        # 组合模式：average/median/first
  
  roi:
    radius: 20                     # ROI区域半径
    edge_weight: 0.1               # 边缘权重
  
  loss:
    lambda_cos: 0.2                # 余弦损失权重
```

---

## ✅ 完整工作流程示例

```bash
# 1. 准备训练数据
python scripts/prepare_training_data.py dataset/training -o processed_data/training.npz

# 2. 验证数据格式
python scripts/verify_npz_format.py processed_data/training.npz

# 3. 训练模型
python scripts/train_stage2.py processed_data/training.npz -o runs/exp1 -e 100

# 4. 准备测试数据
python scripts/prepare_training_data.py dataset/test -o processed_data/test.npz

# 5. 评估
python scripts/evaluate_experiments.py -e runs/exp1 -t processed_data/test.npz -o results/eval.json

# 6. 查看结果
cat results/eval.md
```

---

## 📞 需要帮助？

如遇问题，查看日志：
```bash
# 数据准备日志
cat dataset/training/chip*/debug_*.png

# 训练日志
cat runs/*/training.log
```
