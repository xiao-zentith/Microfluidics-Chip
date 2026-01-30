# 数据集准备与训练指南

> **目标**: 将原始拍摄图像转换为训练数据，并训练 Stage2 光照校正模型

---

## 📁 第一步：准备数据集目录

### 要求的目录结构

```
data/
├── raw/                         # 原始拍摄数据
│   └── microfluidics_v1/       # 数据集名称（可根据实验版本命名）
│       ├── training/           # 训练集目录
│       │   ├── chip001/        # 第1个芯片
│       │   │   ├── gt.png      # Ground Truth
│       │   │   ├── dirty_01.png # 干扰图像
│       │   │   └── ...
│       │   └── chip002/
│       │       └── ...
│       └── test/               # 测试集目录
│           ├── chip003/
│           │   ├── gt.png
│           │   └── ...
│           └── ...
├── processed/                  # 预处理后的NPZ文件
│   └── microfluidics_v1/
│       ├── training.npz
│       └── test.npz
└── experiments/                # 训练输出
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

## 🔧 第二步：运行数据准备脚本

### 基础用法

```bash
python scripts/prepare_training_data.py data/raw/microfluidics_v1/training -o data/processed/microfluidics_v1/training.npz
```

### 使用离线增强 (v1.2)

```bash
# 5倍ISP增强 (推荐)
python scripts/prepare_training_data.py data/raw/microfluidics_v1/training -o data/processed/microfluidics_v1/training.npz \
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
    data/raw/microfluidics_v1/training \
    --output data/processed/microfluidics_v1/training.npz \
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
| `data/processed/microfluidics_v1/training.npz` | 训练数据（target_in, ref_in, labels） |
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

## 🎯 第三步：训练模型

### 日常训练（推荐）

```bash
python scripts/train_stage2.py processed_data/training.npz -o runs/my_training -e 100
```

**参数说明：**
```bash
python scripts/train_stage2.py processed_data/training.npz \
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

## 📊 第四步：评估模型

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

## 🚀 第五步：使用训练好的模型

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
