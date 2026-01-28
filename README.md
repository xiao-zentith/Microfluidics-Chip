# Microfluidics-Chip

> 微流控芯片图像处理流水线 - 基于 YOLO 检测与 UNet 光照校正的自动化分析系统

[![Tests](https://img.shields.io/badge/tests-25%2F25%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📋 项目简介

自动化微流控芯片图像处理系统，包含两阶段流水线：

**Stage 1: 目标检测与几何校正**
- YOLO 目标检测识别 12 个腔室
- 十字几何校正算法实现精准对齐
- 自动切片提取与标准化

**Stage 2: UNet 光照校正**
- 双流 UNet 网络进行光照均匀化
- ROI 加权损失优化核心反应区
- 保留光谱信息的自适应校正

---

## 🚀 快速开始

### 1. 安装

```bash
# 克隆项目
git clone <repository-url>
cd Microfluidics-Chip

# 创建 conda 环境
conda create -n microfluidics python=3.10 -y
conda activate microfluidics

# 安装项目（开发模式）
pip install -e .

# 安装开发依赖（可选）
pip install -e ".[dev]"
```

### 2. 配置权重文件

将训练好的权重文件放置到 `weights/` 目录：

```
weights/
├── yolo/
│   └── best.pt          # YOLO 检测器权重
└── unet/
    └── best_model.pth   # UNet 校正器权重
```

### 3. 运行示例

```bash
# Stage1: 检测与切片
python -m microfluidics_chip.pipelines.cli stage1 \
  data/chip001.png \
  -o runs/stage1

# Stage2: 光照校正
python -m microfluidics_chip.pipelines.cli stage2 \
  runs/stage1/chip001 \
  -o runs/stage2

# 批量处理
python -m microfluidics_chip.pipelines.cli stage1-batch \
  data/images \
  -o runs/batch_stage1

python -m microfluidics_chip.pipelines.cli stage2-batch \
  runs/batch_stage1 \
  -o runs/batch_stage2
```

---

## 📖 详细文档

### CLI 命令

#### Stage1 处理

```bash
# 基本用法
python -m microfluidics_chip.pipelines.cli stage1 IMAGE_PATH -o OUTPUT_DIR

# 带 GT 图像
python -m microfluidics_chip.pipelines.cli stage1 \
  data/chip001.png \
  --gt data/chip001_gt.png \
  -o runs/stage1

# 调试模式（保存检测可视化和单个切片）
python -m microfluidics_chip.pipelines.cli stage1 \
  data/chip001.png \
  -o runs/debug \
  --save-slices \
  --save-debug

# 使用自定义配置
python -m microfluidics_chip.pipelines.cli stage1 \
  data/chip001.png \
  -o runs/stage1 \
  --config configs/my_config.yaml
```

#### Stage2 处理

```bash
# 基本用法（P2 规范：只接受 stage1_run_dir）
python -m microfluidics_chip.pipelines.cli stage2 \
  runs/stage1/chip001 \
  -o runs/stage2

# 批量处理
python -m microfluidics_chip.pipelines.cli stage2-batch \
  runs/stage1 \
  -o runs/stage2
```

### Python API

```python
from pathlib import Path
from microfluidics_chip.core.config import get_default_config
from microfluidics_chip.pipelines.stage1 import run_stage1
from microfluidics_chip.pipelines.stage2 import run_stage2

# 加载配置
config = get_default_config()

# Stage1
stage1_output = run_stage1(
    chip_id="chip001",
    raw_image_path=Path("data/chip001.png"),
    gt_image_path=None,
    output_dir=Path("runs/stage1"),
    config=config.stage1
)

# Stage2
stage2_output = run_stage2(
    stage1_run_dir=Path("runs/stage1/chip001"),
    output_dir=Path("runs/stage2"),
    config=config.stage2
)
```

### 配置文件

创建自定义配置文件 `configs/my_config.yaml`：

```yaml
experiment_name: "my_experiment"

stage1:
  yolo:
    weights_path: "weights/yolo/best.pt"
    confidence_threshold: 0.5
    device: "cuda"
  
  geometry:
    canvas_size: 600
    slice_size: [80, 80]
    crop_radius: 25

stage2:
  weights_path: "weights/unet/best_model.pth"
  model:
    device: "cuda"
    features: [64, 128, 256, 512]
```

---

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/unit/ -v
pytest tests/integration/ -v

# 生成覆盖率报告
pytest tests/ --cov=src/microfluidics_chip --cov-report=html
```

**测试覆盖**: 25/25 tests passing ✅

---

## 📁 项目结构

```
Microfluidics-Chip/
├── src/microfluidics_chip/          # 源代码
│   ├── core/                         # 核心模块（类型、配置、IO）
│   ├── stage1_detection/             # Stage1: YOLO检测+几何校正
│   ├── stage2_correction/            # Stage2: UNet光照校正
│   └── pipelines/                    # 业务编排层+CLI
├── tests/                            # 测试套件
│   ├── unit/                         # 单元测试
│   └── integration/                  # 集成测试
├── configs/                          # 配置文件
├── scripts/                          # 工具脚本
├── weights/                          # 模型权重（不提交）
├── deprecated/                       # 废弃代码（v1.0）
└── docs/                             # 文档

```

---

## 🔧 训练 UNet 模型

### 1. 准备训练数据

#### 数据集结构

训练数据应按以下结构组织：

```
dataset/training/
├── chip001/
│   ├── gt.png          # 理想图（Ground Truth）- 干净无干扰
│   ├── dirty_01.png    # 受干扰图1（距离/光照/角度变化）
│   ├── dirty_02.png    # 受干扰图2
│   ├── dirty_03.png    # 受干扰图3
│   └── ...             # 更多干扰图
├── chip002/
│   ├── gt.png
│   ├── dirty_01.png
│   └── ...
└── chip003/
    └── ...
```

**说明**：
- 每个芯片一个目录
- `gt.png`（或`GT.png`）: 理想图，作为校正目标
- `dirty_*.png`（或`noisy_*.png`）: 受干扰图，每个会生成多条训练数据
- 支持格式：`.png`, `.jpg`, `.jpeg`

#### 数据准备策略

项目支持三种数据准备方式，可根据实际情况选择：

| 策略 | 适用场景 | 数据质量 | 数据量 | 脚本 |
|------|----------|----------|--------|------|
| **真实数据** | 有实际采集数据 | 高（真实场景） | 中 | `prepare_training_data.py` |
| **合成数据** | 真实数据不足 | 中（可控性强） | 大（可无限生成） | `FullChipSynthesizer` |
| **混合数据** | 生产环境（推荐） | 高 | 大 | `prepare_mixed_dataset.py` |

---

#### 方式1：真实数据（1GT + 多Dirty）

处理实际采集的数据，每个芯片包含1张GT和多张干扰图。

```bash
# 数据结构
dataset/real_training/
├── chip001/
│   ├── gt.png
│   ├── dirty_01.png
│   ├── dirty_02.png
│   └── dirty_03.png
└── chip002/
    └── ...

# 生成训练数据
python scripts/prepare_training_data.py \
  dataset/real_training \
  -o data/real_training.npz
```

**特点**：
- ✅ 真实场景数据，泛化能力强
- ✅ 包含真实的噪声和干扰模式
- ⚠️ 需要实际采集，数据量有限

---

#### 方式2：合成数据（1GT × 倍率）

从理想GT图像合成大量训练数据。

```bash
# 数据结构
dataset/clean_images/
├── chip001_clean.png
├── chip002_clean.png
└── ...

# 使用Synthesizer生成
python -c "
from pathlib import Path
from microfluidics_chip.core.config import get_default_config
from microfluidics_chip.stage1_detection.detector import ChamberDetector
from microfluidics_chip.stage1_detection.synthesizer import FullChipSynthesizer
import numpy as np

config = get_default_config()
detector = ChamberDetector(config.stage1.yolo)
synth = FullChipSynthesizer(detector, config.stage1.geometry)

# 运行合成（倍率=50）
synth.run(
    clean_dir=Path('dataset/clean_images'),
    output_path=Path('data/synthetic_training.npz'),
    multiplier=50
)
"
```

**特点**：
- ✅ 可大量生成，数据量充足
- ✅ 可控的干扰参数
- ⚠️ 模拟数据，可能与真实场景有差异

---

#### 方式3：混合数据（推荐）⭐

结合真实数据和合成数据，平衡质量与数量。

```bash
# 生成混合数据集
python scripts/prepare_mixed_dataset.py \
  --real dataset/real_training \
  --synthetic dataset/clean_images \
  -o data/mixed_training.npz \
  --synthetic-multiplier 50

# 仅使用真实数据
python scripts/prepare_mixed_dataset.py \
  --real dataset/real_training \
  -o data/real_only.npz

# 仅使用合成数据
python scripts/prepare_mixed_dataset.py \
  --synthetic dataset/clean_images \
  -o data/synthetic_only.npz \
  --synthetic-multiplier 100
```

**输出示例**：
```
Dataset Composition:
  - Real data:      270 samples (10%)
  - Synthetic data: 2430 samples (90%)
  - Total:          2700 samples
```

**推荐配置**：
- 小规模：10芯片真实数据 + 5张GT×50倍 ≈ 3000样本
- 中规模：50芯片真实数据 + 20张GT×50倍 ≈ 12000样本
- 大规模：100芯片真实数据 + 50张GT×100倍 ≈ 60000样本

---

#### 数据准备流程

Stage1 会对每个芯片执行以下处理：

1. **检测腔室**: GT图 + 每个Dirty图 → YOLO检测 → 12个腔室位置
2. **几何校正**: 基于检测位置进行变换对齐
3. **切片提取**: 提取12个腔室切片
4. **配对**: 每个Dirty腔室 ↔ 对应GT腔室
5. **基准提取**: 提取3个基准腔室(索引0-2)用于UNet双流输入

**处理命令**：

```bash
# 准备训练数据
python scripts/prepare_training_data.py \
  dataset/training \
  -o data/training.npz

# 使用自定义配置
python scripts/prepare_training_data.py \
  dataset/training \
  -o data/training.npz \
  --config configs/my_config.yaml

# 不保存调试图像
python scripts/prepare_training_data.py \
  dataset/training \
  -o data/training.npz \
  --no-debug
```

**输出**：
```
data/training.npz
├── signals    # (N, H, W, 3) 干扰腔室切片
├── references # (N, 3, H, W, 3) 3个基准腔室切片
└── targets    # (N, H, W, 3) 理想腔室切片（GT）
```

**调试输出** (在每个芯片目录):
```
dataset/training/chip001/
├── debug_gt.png           # GT检测可视化
├── debug_dirty_01.png     # Dirty_01检测可视化
└── ...
```

**数据集统计示例**:
```
Dataset Statistics:
  - Total chips: 10
  - Total samples: 810
  - Avg samples/chip: 81.0
```

**说明**: 
- 每个Dirty图 × 每个腔室(9个，跳过3个基准) = 9条样本
- 3个Dirty图 × 9个腔室 = 27条样本/芯片
- 10个芯片 × 27 = 270条样本（最小示例）

---

### 2. 训练模型

```bash
# 训练 Stage2 UNet
python scripts/train_stage2.py \
  data/synthetic_data.npz \
  -o runs/training \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4

# 使用自定义参数
python scripts/train_stage2.py \
  data/synthetic_data.npz \
  -o runs/training \
  --epochs 200 \
  --roi-radius 20 \
  --lambda-cos 0.2 \
  --device cuda
```

---

## 📊 输出格式

### Stage1 输出

```
runs/stage1/chip001/
├── stage1_metadata.json      # 元数据（检测数、处理时间等）
├── aligned.png                # 对齐后的完整图像
├── chamber_slices.npz         # 12个切片（key='slices'）
├── debug_detection.png        # 检测可视化（可选）
└── slices/                    # 单个切片图像（可选）
    ├── 0_raw.jpg
    ├── 1_raw.jpg
    └── ...
```

### Stage2 输出

```
runs/stage2/chip001/
├── stage2_metadata.json       # 元数据
└── corrected_slices.npz       # 校正后的切片（key='slices'）
```

---

## 🛠️ 故障排查

### 1. YOLO 权重文件找不到

**错误**: `FileNotFoundError: weights/yolo/best.pt`

**解决**: 
- 确保权重文件在 `weights/yolo/best.pt`
- 或修改 `configs/default.yaml` 中的 `weights_path`

### 2. OpenMP 库冲突警告

**错误**: `OMP: Error #15: Initializing libiomp5md.dll`

**解决**:
```bash
# 设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac
set KMP_DUPLICATE_LIB_OK=TRUE     # Windows
```

### 3. CUDA 不可用

**错误**: `cuda not available`

**解决**: 修改配置文件，将 `device: "cuda"` 改为 `device: "cpu"`

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

**代码规范**:
```bash
# 格式化代码
black src/ tests/

# 类型检查
mypy src/

# Lint 检查
ruff check src/
```

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 📧 联系方式

- **项目维护**: Microfluidics Team
- **问题反馈**: [Issues](../../issues)

---

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 目标检测框架
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [OpenCV](https://opencv.org/) - 图像处理库
