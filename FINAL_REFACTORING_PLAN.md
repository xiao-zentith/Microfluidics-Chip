# 微流控芯片项目 - 完整重构方案

> **项目目标**: 将Microfluidics-Chip项目重构为生产级、可维护、可复现、易部署的科研代码库  
> **核心原则**: 职责单一、接口固化、CLI First、完整追溯、支持消融/对比实验

---

## 📑 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [当前代码分析](#2-当前代码分析)
3. [最终目录结构](#3-最终目录结构)
4. [核心设计决策](#4-核心设计决策)
5. [数据契约（强类型）](#5-数据契约强类型)
6. [核心组件实现](#6-核心组件实现)
7. [实验框架设计](#7-实验框架设计)
8. [配置管理系统](#8-配置管理系统)
9. [文件迁移映射](#9-文件迁移映射)
10. [分步实施计划](#10-分步实施计划)
11. [测试策略](#11-测试策略)
12. [部署指南](#12-部署指南)

---

## 1. 项目背景与目标

### 1.1 三阶段处理Pipeline

本项目实现微流控芯片图像的自动化分析：

**Stage 1: 目标检测与几何校正**
- YOLO检测12个腔室中心点
- 基于芯片重心的拓扑排序
- 刚性变换（旋转+缩放+平移）对齐到理想布局
- 提取12个固定尺寸的切片

**Stage 2: 光照校正（双流UNet）**
- 输入：Stage1的切片（Target + Reference）
- 模型：双流UNet（Signal Stream + Reference Encoder）
- 输出：校正后的干净切片

**Stage 3: 浓度提取（预留）**
- 从复原的RGB图像提取浓度值
- 线性回归或端到端模型

### 1.2 重构目标

✅ **生产级架构**: Src-Layout + pip可安装  
✅ **强类型接口**: Pydantic数据契约  
✅ **配置管理**: YAML + 环境变量  
✅ **实验追溯**: Manifest + Git追踪  
✅ **支持实验**: 消融/对比实验框架  
✅ **跨平台**: 本地Windows开发 + 远程Linux GPU部署

---

## 2. 当前代码分析

### 2.1 真实Pipeline（需迁移）

| 当前路径 | 功能 | 迁移目标 |
|---------|------|---------|
| `preprocess/detector.py` | YOLO检测器 | `stage1_detection/detector.py` |
| `preprocess/utils.py` | CrossGeometryEngine（核心） | `stage1_detection/geometry_engine.py` |
| `preprocess/pipeline.py` | Stage1流水线 | `stage1_detection/inference.py` |
| `preprocess/synthesizer_chip.py` | 数据增强 | `stage1_detection/synthesizer.py` |
| `preprocess/main.py` | 批处理脚本 | `scripts/data_preparation/batch_process.py` |
| `unet/model/unet.py` | 双流UNet + Loss | `stage2_correction/models/` + `losses.py` |
| `unet/model/train.py` | 训练代码 | `stage2_correction/trainer.py` + `scripts/training/` |

### 2.2 Demo代码（需废弃）

以下代码未被实际pipeline引用，为早期demo：

- `gpc/` - GPC分类demo
- `correction/` - 光照校正demo
- `match/` - 配对匹配demo
- `preprocess/synthesizer.py` - 旧版增强
- `unet/augmentation/` - 高斯blob增强demo

**处理方式**: 移至 `deprecated/` 目录

---

## 3. 最终目录结构

```
Microfluidics-Chip/
├── src/
│   └── microfluidics_chip/
│       ├── __init__.py
│       │
│       ├── stage1_detection/              # 阶段1：检测与几何校正
│       │   ├── __init__.py
│       │   ├── detector.py                # YOLO检测器（纯算法）
│       │   ├── geometry_engine.py         # 几何引擎（含切片）
│       │   ├── inference.py               # 推理入口（返回Result）
│       │   └── synthesizer.py             # 数据增强
│       │
│       ├── stage2_correction/             # 阶段2：UNet光照校正
│       │   ├── __init__.py
│       │   ├── models/
│       │   │   ├── __init__.py
│       │   │   ├── dual_stream_unet.py    # 双流UNet（Ours）
│       │   │   └── single_stream_unet.py  # 单流baseline（消融A）
│       │   ├── losses.py                  # ROIWeightedLoss
│       │   ├── dataset.py                 # Dataset类
│       │   ├── trainer.py                 # Trainer类（无main）
│       │   └── inference.py               # 推理入口
│       │
│       ├── stage3_concentration/          # 阶段3：浓度提取
│       │   ├── __init__.py
│       │   ├── models/
│       │   │   ├── __init__.py
│       │   │   └── end_to_end_regressor.py # E2E baseline（对比实验）
│       │   ├── rgb_extractor.py
│       │   └── concentration_calculator.py
│       │
│       ├── core/                          # 核心公共模块
│       │   ├── __init__.py
│       │   ├── config.py                  # Pydantic配置定义
│       │   ├── config_loader.py           # 配置加载器
│       │   ├── types.py                   # 强类型数据契约
│       │   ├── io.py                      # ResultSaver统一IO
│       │   ├── manifest.py                # 实验追溯
│       │   ├── experiment_manager.py      # 实验管理器
│       │   ├── logger.py                  # 日志系统
│       │   ├── metrics.py                 # 评估指标
│       │   └── exceptions.py              # 自定义异常
│       │
│       └── pipelines/                     # 唯一业务编排入口
│           ├── __init__.py
│           ├── stage1.py                  # Stage1编排（批处理+IO）
│           ├── stage2.py                  # Stage2编排
│           ├── full.py                    # 完整流水线
│           └── cli.py                     # CLI统一入口
│
├── scripts/                               # 独立执行脚本
│   ├── training/
│   │   ├── train_stage1_yolo.py           # YOLO训练
│   │   ├── train_stage2_dual.py           # 双流训练
│   │   ├── train_stage2_single.py         # 单流训练
│   │   └── train_stage3_e2e.py            # E2E训练
│   ├── data_preparation/
│   │   ├── batch_process_stage1.py        # 批量Stage1
│   │   ├── generate_synthetic_data.py     # 运行synthesizer
│   │   └── split_dataset.py
│   └── visualization/
│       ├── visualize_yolo.py
│       └── visualize_geometry.py
│
├── experiments/                           # 实验分析（仅后处理）
│   ├── notebooks/
│   │   ├── ablation_a_analysis.ipynb      # 消融A分析
│   │   ├── ablation_b_analysis.ipynb      # 消融B分析
│   │   └── comparison_visualization.ipynb # 对比实验可视化
│   └── plotting/
│       ├── plot_ablation_comparison.py
│       └── generate_paper_figures.py
│
├── configs/
│   ├── default.yaml                       # 默认配置
│   ├── env/
│   │   ├── local.yaml                     # 本地环境
│   │   └── remote.yaml                    # 远程GPU环境
│   └── experiments/
│       ├── ablation_a_dual.yaml           # 消融A: 双流
│       ├── ablation_a_single.yaml         # 消融A: 单流
│       ├── ablation_b_synthetic.yaml      # 消融B: 合成数据
│       ├── ablation_b_real.yaml           # 消融B: 真实数据
│       └── comparison_e2e.yaml            # 对比: E2E
│
├── data/                                  # gitignored
├── runs/                                  # gitignored
│   └── {timestamp}_{exp_name}/
│       ├── manifest.json
│       ├── config_resolved.yaml
│       ├── weights/
│       ├── metrics.json
│       └── artifacts/
├── weights/                               # gitignored
│
├── tests/
│   ├── unit/
│   │   ├── test_types_serialization.py
│   │   └── test_geometry_engine.py
│   └── integration/
│       ├── test_stage1_smoke.py
│       └── test_stage1_smoke_mocked.py    # Mock测试（不依赖权重）
│
├── deprecated/                            # 完全隔离
│   ├── gpc/
│   ├── correction/
│   ├── match/
│   ├── preprocess/synthesizer.py
│   └── unet/augmentation/
│
├── pyproject.toml
├── pytest.ini
├── .gitignore
└── README.md
```

---

## 4. 核心设计决策

### 4.1 职责分离：inference vs pipelines

**问题**: 原`pipeline.py`混杂算法和IO，职责不清

**解决**:
- **`stageX/inference.py`**: 纯算法，输入/输出内存对象（np.ndarray），无IO
- **`pipelines/stageX.py`**: 业务编排，负责文件读写、批处理、日志

**优势**:
- ✅ 可测试性：inference可直接用numpy测试
- ✅ 可复用性：其他项目可只导入inference
- ✅ 清晰边界：算法与业务完全分离

### 4.2 双层数据对象：Result vs Output

**问题**: 混用内存对象和落盘路径，导致序列化失败

**解决**:
- **`StageXResult`**: 内存对象，包含np.ndarray，用于inference返回
- **`StageXOutput`**: 落盘对象（DTO），仅路径+元数据，用于保存

**示例**:
```python
# inference返回Result（内存）
result: Stage1Result = infer_stage1(image, chip_id, config)
# aligned_image: np.ndarray

# pipelines保存后转为Output（落盘）
output: Stage1Output = stage1_result_to_output(result, ...)
# aligned_image_path: Path("aligned.png")
```

### 4.3 固定文件命名与相对路径

**规范**:

**Stage1输出目录**:
```
{run_dir}/
├── stage1_metadata.json       # 固定
├── chamber_slices.npz          # 固定，key="slices"
├── aligned.png                 # 固定
└── debug_visualization.png     # 固定
```

**Stage2输出目录**:
```
{run_dir}/
├── stage2_metadata.json       # 固定
└── corrected_slices.npz        # 固定，key="slices"
```

**路径存储**: metadata中所有路径字段存储相对路径
```json
{
  "aligned_image_path": "aligned.png",  // 相对路径
  "chamber_slices_path": "chamber_slices.npz"
}
```

**加载时解析**: `absolute_path = run_dir / Path(rel_path)`

### 4.4 CLI First统一入口

**唯一入口**: `python -m microfluidics_chip.pipelines.cli`

**子命令**:
```bash
# Stage1处理
cli stage1 image.jpg --config default.yaml --output runs/test

# Stage2处理
cli stage2 --stage1-run-dir runs/stage1_chip001 --output runs/test

# 训练
cli train --stage 2 --config experiments/ablation_a_dual.yaml

# 评估
cli evaluate --experiment ablation_a_dual --baseline ablation_a_single
```

### 4.5 实验追溯（ExperimentManager + Manifest）

**每次运行自动记录**:
- Git commit hash（含dirty状态）
- 完整配置快照（config_resolved.yaml）
- 模型权重SHA256
- 运行环境（Python/Torch/CUDA版本、GPU型号）

**标准化输出**:
```
runs/{timestamp}_{exp_name}/
├── manifest.json          # 完整追溯信息
├── config_resolved.yaml   # 最终生效配置
├── weights/               # 固化的权重文件
├── metrics.json           # 最终指标
└── artifacts/             # 可视化图表
```

---

## 5. 数据契约（强类型）

### 5.1 基础类型

```python
# src/microfluidics_chip/core/types.py

from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

class ChamberDetection(BaseModel):
    """单个腔室检测结果"""
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    class_name: str  # "chamber_blank" 或 "chamber_lit"
    confidence: float

class TransformParams(BaseModel):
    """几何变换参数"""
    rotation_matrix: List[List[float]]
    rotation_angle: float
    chip_centroid: Tuple[float, float]
    blank_arm_index: int
```

### 5.2 Stage1数据契约

```python
class Stage1Result(BaseModel):
    """Stage1内存结果（inference返回）"""
    chip_id: str
    aligned_image: Any  # np.ndarray
    chamber_slices: Any  # np.ndarray (12×H×W×3)
    transform_params: TransformParams
    chambers: List[ChamberDetection]
    gt_slices: Optional[Any] = None
    debug_vis: Optional[Any] = None
    processing_time: float = 0.0

class Stage1Output(BaseModel):
    """Stage1落盘结果（pipelines保存）"""
    chip_id: str
    aligned_image_path: Path  # 相对路径："aligned.png"
    chamber_slices_path: Path  # 相对路径："chamber_slices.npz"
    transform_params: TransformParams
    num_chambers: int
    debug_vis_path: Optional[Path] = None
    processing_time: float = 0.0
```

### 5.3 Stage2数据契约

```python
class Stage2Result(BaseModel):
    """Stage2内存结果"""
    chip_id: str
    corrected_slices: Any  # np.ndarray
    correction_params: Dict[str, List[float]]
    metrics: Optional[Dict[str, float]] = None
    processing_time: float = 0.0

class Stage2Output(BaseModel):
    """Stage2落盘结果"""
    chip_id: str
    corrected_slices_path: Path  # 相对路径："corrected_slices.npz"
    correction_params: Dict[str, List[float]]
    metrics: Optional[Dict[str, float]] = None
    processing_time: float = 0.0
```

---

## 6. 核心组件实现

### 6.1 Stage1调用链

```
CLI入口
└── pipelines/cli.py::stage1()
    └── pipelines/stage1.py::run_stage1()
        ├── cv2.imread()                    # 读取图像
        └── stage1_detection/inference.py::infer_stage1()
            ├── detector.py::ChamberDetector.detect()
            │   └── YOLO推理 → List[ChamberDetection]
            │
            └── geometry_engine.py::CrossGeometryEngine.process()
                ├── _topological_sort()     # 极坐标拓扑排序
                ├── _find_blank_arm()       # 锚点定位
                ├── _compute_rigid_matrix() # 计算变换矩阵
                ├── cv2.warpAffine()        # 图像对齐
                └── _extract_slices()       # 切片提取（内置）
```

**关键说明**:
- `CrossGeometryEngine.process()` **已包含切片逻辑**，不需要单独的slicer
- 原`preprocess/main.py`的批处理逻辑迁移到`pipelines/stage1.py::run_stage1_batch()`

### 6.2 批处理性能优化

**依赖注入模式**:

```python
# pipelines/stage1.py::run_stage1_batch()

# 循环外初始化（避免重复加载YOLO）
detector = ChamberDetector(config.yolo)
geometry_engine = CrossGeometryEngine(config.geometry)

for img_path in image_files:
    # 循环内复用实例
    result = infer_stage1(
        image, chip_id, config,
        detector=detector,  # 传入实例
        geometry_engine=geometry_engine
    )
```

### 6.3 Stage2加载Stage1产物

```python
# pipelines/stage2.py

def load_stage1_output(stage1_run_dir: Path) -> Tuple[Stage1Output, np.ndarray]:
    """从固定文件名加载"""
    # 1. 加载metadata（固定文件名）
    metadata_path = stage1_run_dir / "stage1_metadata.json"
    with open(metadata_path) as f:
        stage1_output = Stage1Output(**json.load(f))
    
    # 2. 解析相对路径
    slices_abs_path = stage1_run_dir / stage1_output.chamber_slices_path
    
    # 3. 加载npz（固定key="slices"）
    slices_data = np.load(slices_abs_path)
    chamber_slices = slices_data['slices']
    
    return stage1_output, chamber_slices
```

---

## 7. 实验框架设计

### 7.1 消融实验A：双流 vs 单流

**目的**: 证明Reference Stream的必要性

| 方法 | 模型 | 配置 |
|------|------|------|
| Ours | RefGuidedUNet（双流） | `ablation_a_dual.yaml` |
| Baseline | SingleStreamUNet（单流） | `ablation_a_single.yaml` |

**运行**:
```bash
# 训练双流
cli train --stage 2 --config experiments/ablation_a_dual.yaml

# 训练单流
cli train --stage 2 --config experiments/ablation_a_single.yaml

# 对比评估
cli evaluate --experiment ablation_a_dual --baseline ablation_a_single
```

### 7.2 消融实验B：合成数据 vs 真实数据

**目的**: 证明Sim-to-Real数据合成的价值

| 方法 | 数据集 | 规模 |
|------|--------|------|
| Ours | 合成数据 | 3000组（synthesizer_chip.py） |
| Baseline | 真实数据 | 77组（7张×11切片） |

### 7.3 对比实验：复原 vs 端到端

**目的**: 证明复原方案优于黑盒回归

| 方法 | 流程 | 可解释性 |
|------|------|---------|
| Ours | Stage1→Stage2→Stage3（复原+RGB提取） | ⭐⭐⭐⭐⭐ |
| Baseline | End2EndRegressor（ResNet直接回归） | ⭐ |

---

## 8. 配置管理系统

### 8.1 配置合并优先级

```
Priority 1 (Base)   : configs/default.yaml
Priority 2 (Env)    : configs/env/{local|remote}.yaml
Priority 3 (Exp)    : configs/experiments/xxx.yaml
Priority 4 (CLI)    : 命令行overrides
```

**最终配置**: 保存为`runs/{id}/config_resolved.yaml`

### 8.2 配置示例

```yaml
# configs/default.yaml
experiment_name: "microfluidics_stage1"

paths:
  data_dir: "data"
  runs_dir: "runs"
  weights_dir: "weights"

stage1:
  yolo:
    weights_path: "weights/yolo/best.pt"
    confidence_threshold: 0.5
    device: "cuda"
  
  geometry:
    canvas_size: 600
    slice_size: [80, 80]
    ideal_center_gap: 60
    ideal_chamber_step: 50
```

### 8.3 环境变量支持

```yaml
# configs/env/remote.yaml
paths:
  data_dir: "${MICROFLUIDICS_DATA_DIR}"  # 环境变量
  weights_dir: "/mnt/shared/weights"

stage1:
  yolo:
    device: "cuda"
```

---

## 9. 文件迁移映射

### 9.1 Stage1迁移

| 当前路径 | 新路径 | 备注 |
|---------|--------|------|
| `preprocess/detector.py` | `src/microfluidics_chip/stage1_detection/detector.py` | 添加配置注入 |
| `preprocess/utils.py` | `src/microfluidics_chip/stage1_detection/geometry_engine.py` | 保持CrossGeometryEngine类 |
| `preprocess/pipeline.py` | `src/microfluidics_chip/stage1_detection/inference.py` | 改为返回Result |
| `preprocess/synthesizer_chip.py` | `src/microfluidics_chip/stage1_detection/synthesizer.py` | 适配新接口 |
| `preprocess/main.py` | `scripts/data_preparation/batch_process_stage1.py` | 批处理脚本 |
| `preprocess/config.py` | `configs/default.yaml` | 转为YAML |

### 9.2 Stage2迁移

| 当前路径 | 新路径 |
|---------|--------|
| `unet/model/unet.py` (RefGuidedUNet) | `src/microfluidics_chip/stage2_correction/models/dual_stream_unet.py` |
| `unet/model/unet.py` (ROIWeightedLoss) | `src/microfluidics_chip/stage2_correction/losses.py` |
| `unet/model/train.py` (Dataset) | `src/microfluidics_chip/stage2_correction/dataset.py` |
| `unet/model/train.py` (训练循环) | `src/microfluidics_chip/stage2_correction/trainer.py` |

### 9.3 废弃代码

| 当前路径 | 新路径 |
|---------|--------|
| `gpc/` | `deprecated/gpc/` |
| `correction/` | `deprecated/correction/` |
| `match/` | `deprecated/match/` |
| `preprocess/synthesizer.py` | `deprecated/preprocess/synthesizer.py` |
| `unet/augmentation/` | `deprecated/unet/augmentation/` |

---

## 10. 分步实施计划

### Step 1: 基础设施搭建（1-2天）

**目标**: 创建目录结构、配置系统、核心模块

**文件**:
```
✓ 创建src/microfluidics_chip/目录结构
✓ pyproject.toml
✓ configs/default.yaml
✓ core/config.py, types.py, io.py, logger.py
✓ core/manifest.py, experiment_manager.py
```

**验证**:
```bash
pip install -e .
python -c "import microfluidics_chip; print('✓')"
```

### Step 2: Stage1迁移（2-3天）

**目标**: 迁移YOLO检测器和几何引擎

**文件**:
```
✓ stage1_detection/detector.py
✓ stage1_detection/geometry_engine.py
✓ stage1_detection/inference.py
✓ pipelines/stage1.py
```

**验证**:
```bash
pytest tests/unit/test_geometry_engine.py
pytest tests/integration/test_stage1_smoke_mocked.py
```

### Step 3: Stage2迁移（2-3天）

**目标**: 迁移UNet模型和训练代码

**文件**:
```
✓ stage2_correction/models/dual_stream_unet.py
✓ stage2_correction/losses.py
✓ stage2_correction/dataset.py
✓ stage2_correction/trainer.py
✓ stage2_correction/inference.py
✓ pipelines/stage2.py
```

**验证**:
```bash
pytest tests/unit/test_stage2_model.py
```

### Step 4: CLI和实验框架（1-2天）

**目标**: 统一入口和实验管理

**文件**:
```
✓ pipelines/cli.py
✓ configs/experiments/*.yaml
✓ experiments/plotting/*.py
```

**验证**:
```bash
cli --help
cli stage1 --help
```

### Step 5: 测试和文档（1-2天）

**目标**: 完整测试覆盖和文档

**文件**:
```
✓ tests/unit/
✓ tests/integration/
✓ README.md
✓ docs/
```

**验证**:
```bash
pytest tests/ -v --cov=microfluidics_chip
```

---

## 11. 测试策略

### 11.1 单元测试

```
tests/unit/
├── test_types_serialization.py    # 验证Output可JSON序列化
├── test_geometry_engine.py         # 验证几何变换逻辑
├── test_config_loader.py           # 验证配置合并
└── test_result_saver.py            # 验证IO操作
```

### 11.2 集成测试（Mock）

```python
# tests/integration/test_stage1_smoke_mocked.py
# 完全不依赖YOLO权重，使用Mock

with patch('ChamberDetector') as MockDetector:
    mock_detector.detect.return_value = mock_chambers
    
    output = run_stage1(image_path, config, output_dir)
    
    # 验证固定文件名
    assert (output_dir / "stage1_metadata.json").exists()
    assert (output_dir / "chamber_slices.npz").exists()
    
    # 验证npz key
    data = np.load(output_dir / "chamber_slices.npz")
    assert 'slices' in data
```

### 11.3 集成测试（真实）

```
tests/integration/
├── test_stage1_full.py             # 需要YOLO权重
├── test_stage2_full.py             # 需要UNet权重
└── test_full_pipeline.py           # 端到端测试
```

---

## 12. 部署指南

### 12.1 本地开发（Windows）

```bash
# 1. 安装
git clone <repo>
cd Microfluidics-Chip
pip install -e ".[dev]"

# 2. 配置环境
cp configs/env/local.yaml.example configs/env/local.yaml
# 编辑local.yaml，设置本地路径

# 3. 运行示例
python -m microfluidics_chip.pipelines.cli stage1 \
    data/raw/chip_001.jpg \
    --env local \
    --output runs/test
```

### 12.2 远程GPU部署（Linux）

```bash
# 1. SSH到服务器
ssh user@gpu-server

# 2. 克隆并安装
git clone <repo>
cd Microfluidics-Chip
pip install -e .

# 3. 配置环境
export MICROFLUIDICS_DATA_DIR=/mnt/shared/data
export MICROFLUIDICS_WEIGHTS_DIR=/mnt/shared/weights

# 4. 运行训练
python -m microfluidics_chip.pipelines.cli train \
    --stage 2 \
    --config experiments/ablation_a_dual.yaml \
    --env remote
```

### 12.3 pyproject.toml配置

```toml
[project]
name = "microfluidics-chip"
version = "0.1.0"
dependencies = [
    "torch>=2.0",
    "ultralytics>=8.0",
    "opencv-python>=4.8",
    "numpy>=1.24",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "PyYAML>=6.0",
    "gitpython>=3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
norecursedirs = ["deprecated", ".git", "dist", "build"]

[tool.black]
line-length = 100
exclude = "deprecated/"

[tool.ruff]
line-length = 100
exclude = ["deprecated/"]
```

---

## 📊 重构效果预期

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| **代码组织** | 文件散落根目录 | Src-Layout标准结构 |
| **配置管理** | 硬编码路径 | YAML+环境变量 |
| **可测试性** | 依赖权重文件 | Mock测试1秒完成 |
| **实验追溯** | 手动记录 | 自动Manifest |
| **批处理性能** | 每图重新加载YOLO | 循环外初始化 |
| **跨环境部署** | 手动改路径 | 环境配置自动切换 |
| **消融实验** | 需手动组织代码 | 配置文件+CLI即可 |

---

## ✅ 验证清单

重构完成后，运行以下命令验证：

```bash
# 1. 安装验证
pip install -e ".[dev]"
python -c "import microfluidics_chip; print('✓ 导入成功')"

# 2. Mock测试（不需要权重）
pytest tests/integration/test_stage1_smoke_mocked.py -v
# 预期：PASSED，耗时<1秒

# 3. 类型序列化测试
pytest tests/unit/test_types_serialization.py -v
# 预期：验证Output无numpy数组

# 4. CLI测试
python -m microfluidics_chip.pipelines.cli --help
python -m microfluidics_chip.pipelines.cli stage1 --help

# 5. 配置加载测试
pytest tests/unit/test_config_loader.py -v
# 预期：验证4级优先级合并

# 6. 文件命名验证
# 运行Stage1后检查：
ls runs/test_chip/
# 应显示：stage1_metadata.json, chamber_slices.npz, aligned.png

# 7. 相对路径验证
cat runs/test_chip/stage1_metadata.json | grep "_path"
# 应显示："aligned.png"（相对路径）

# 8. NPZ key验证
python -c "import numpy as np; data=np.load('runs/test_chip/chamber_slices.npz'); print(list(data.keys()))"
# 应显示：['slices']
```

---

## 🎯 下一步行动

1. **Review本方案**：确认各部分设计符合需求
2. **创建Git分支**：`git checkout -b refactoring/v1.0`
3. **执行Step 1**：基础设施搭建
4. **逐步迁移**：按Step 2-5顺序执行
5. **持续测试**：每个Step完成后运行对应测试

---

**方案版本**: v1.0  
**创建日期**: 2026-01-27  
**状态**: Ready for Implementation
