# Microfluidics-Chip v1.2 - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.0] - 2026-02-05

### 🎯 Adaptive Detection Pipeline

全新自适应检测管线，解决暗腔室漏检、远近尺度变化和复杂光照环境下的检测问题。

### ✨ Added

#### 数据与标签策略
- **标签迁移脚本** (`scripts/migrate_labels_to_single_class.py`): 多类别→单类别迁移
  - 支持 dry-run 预览、自动备份、更新 data.yaml
  
- **分层增强脚本** (`scripts/augment_yolo_dataset.py` 改造):
  - 分层采样: 70% mild / 25% medium / 5% extreme
  - CLAHE/Invert 开关 (`--enable-clahe`, `--no-invert`)

#### Stage1 自适应检测
- **预处理模块** (`stage1_detection/preprocess.py`):
  - `apply_clahe()`: LAB L通道对比度增强
  - `apply_invert()`: 亮度反转
  - `preprocess_image()`: 统一预处理流水线

- **自适应检测器** (`stage1_detection/adaptive_detector.py`):
  - 粗到精检测: global_scan → cluster_roi → fine_scan
  - DBSCAN 聚类自动 ROI
  - 坐标映射回原图

- **拓扑拟合器** (`stage1_detection/topology_fitter.py`):
  - 十字模板定义 (4臂×3腔室，无中心)
  - RANSAC Similarity Transform 拟合
  - 缺失腔室回填
  - 暗腔室亮度判定 (位于臂最外侧)

- **集成入口** (`stage1_detection/inference.py`):
  - 新增 `infer_stage1_adaptive()` 函数

#### 配置与类型
- **新增配置类** (`core/config.py`):
  - `AdaptiveDetectionConfig`: 粗细扫描、聚类参数
  - `TopologyConfig`: 模板、RANSAC、亮度判定参数

- **新增类型** (`core/types.py`):
  - `AdaptiveDetectionResult`: 完整检测结果

#### 示例与配置模板
- **端到端示例** (`examples/adaptive_detection_demo.py`)
- **配置模板** (`configs/adaptive_detection.yaml`)

#### 测试
- **单元测试** (`tests/unit/test_adaptive_detection.py`): 14 tests passed

---


## [1.1.0] - 2026-01-28

### 🎯 Major Refactoring

Complete project restructuring following v1.1 architecture with strict adherence to P0-P4 constraints.

### ✨ Added

#### Core Infrastructure
- **Type System** (`core/types.py`): Pydantic-based data models with JSON serialization
  - `ChamberDetection`: Detection results (P0 interface)
  - `Stage1Output`, `Stage2Output`: Pipeline outputs with relative paths (P1)
  - `TransformParams`: Geometry transformation parameters
  
- **Configuration System** (`core/config.py`): YAML-based with Pydantic validation
  - `YOLOConfig`, `GeometryConfig`: Stage1 configurations
  - `UNetModelConfig`, `ROILossConfig`: Stage2 configurations
  - Configuration merging and environment variable support

- **IO System** (`core/io.py`): Fixed file naming (P2)
  - `ResultSaver`: Unified save/load interface
  - `save_stage1_result()`, `load_stage1_output()`: Fixed filenames
  - `save_stage2_result()`, `load_stage2_output()`: Fixed filenames
  - NPZ files with unified key naming (`key='slices'`)

- **Logging** (`core/logger.py`): Rich-based unified logging system

#### Stage1: Detection & Geometry
- **Detector** (`stage1_detection/detector.py`): YOLO chamber detector
  - Returns `List[ChamberDetection]` (P0 interface)
  
- **Geometry Engine** (`stage1_detection/geometry_engine.py`): Cross-geometry correction
  - Returns 4-tuple: `(aligned_image, chamber_slices, transform_params, debug_vis)` (P0)
  - Preserves v1.0 algorithms: Real-Coordinate Following, Identity Following, Centroid Sorting
  
- **Synthesizer** (`stage1_detection/synthesizer.py`): Full-chip synthesizer
  - Migrated from v1.0 with algorithm preservation
  
- **Inference** (`stage1_detection/inference.py`): Stage1 inference entry point
  - GT isolation: Separate engine instance for GT processing (P3)

- **Pipeline** (`pipelines/stage1.py`): Stage1 orchestration
  - Fixed file naming (P2)
  - Batch optimization: Models initialized outside loop (P4)
  - Debug visualization support

#### Stage2: Correction
- **Dual-Stream UNet** (`stage2_correction/models/dual_stream_unet.py`)
  - 100% architecture preservation from v1.0
  - Signal encoder + Reference encoder + Fusion + Decoder
  
- **ROI Weighted Loss** (`stage2_correction/losses.py`)
  - ROI-weighted MSE loss for photometric accuracy
  - Cosine similarity loss for spectral accuracy
  - Dynamic weight map generation
  
- **Dataset** (`stage2_correction/dataset.py`)
  - Load synthetic data from NPZ files
  - Auto train/val split
  
- **Trainer** (`stage2_correction/trainer.py`)
  - Training loop with AdamW + LR scheduling
  - Validation with PSNR metrics
  - Paper-level visualization (5-panel comparison)
  
- **Inference** (`stage2_correction/inference.py`): Stage2 inference entry point
  - Dependency injection for batch processing
  
- **Pipeline** (`pipelines/stage2.py`): Stage2 orchestration
  - P2: Only accepts `stage1_run_dir` parameter
  - Batch optimization with model reuse

#### CLI & Scripts
- **Typer CLI** (`pipelines/cli.py`): Unified command-line interface
  - `stage1`: Single chip processing
  - `stage1-batch`: Batch processing
  - `stage2`: Stage2 processing (P2: stage1_run_dir only)
  - `stage2-batch`: Batch Stage2
  - Rich output formatting
  - Debug visualization options
  
- **Training Script** (`scripts/train_stage2.py`): UNet training
  - Command-line parameter configuration
  - Automatic logging and checkpointing

#### Testing
- **Unit Tests** (`tests/unit/`):
  - Type serialization tests (P0, P1 validation)
  - ResultSaver tests (P2 validation)
  
- **Integration Tests** (`tests/integration/`):
  - Stage1 smoke tests (P0, P1, P2 validation, mocked)
  - Stage2 smoke tests (P1, P2 validation, model structure)
  
- **Test Coverage**: 25/25 tests passing ✅

#### Documentation
- **README.md**: Complete project documentation
  - Installation guide
  - CLI usage examples
  - Python API examples
  - Configuration guide
  - Troubleshooting
  
- **Examples** (`examples/end_to_end.py`): Python API usage examples
- **CHANGELOG.md**: This file

### 🔒 Constraints (P0-P4)

**P0 [Interface Locking]**: ✅ Enforced
- `ChamberDetector.detect()` → `List[ChamberDetection]`
- `CrossGeometryEngine.process()` → `(aligned_image, chamber_slices, transform_params, debug_vis)`

**P1 [Path Type]**: ✅ Enforced
- All paths in `StageXOutput` are `str` (relative paths)
- No `pathlib.Path` in output DTOs

**P2 [Fixed Naming]**: ✅ Enforced
- Stage1: `stage1_metadata.json`, `aligned.png`, `chamber_slices.npz` (key='slices')
- Stage2: `stage2_metadata.json`, `corrected_slices.npz` (key='slices')
- No glob/fuzzy matching in loaders

**P3 [GT Isolation]**: ✅ Implemented
- `infer_stage1()` creates separate `CrossGeometryEngine` for GT processing

**P4 [Batch Optimization]**: ✅ Implemented
- `run_stage1_batch()` initializes models outside loop
- `run_stage2_batch()` reuses model instance

### 🔧 Changed

- **Project Structure**: Restructured to `src/` layout with proper packaging
- **Configuration**: Migrated from hardcoded constants to YAML + Pydantic
- **Logging**: Unified logging with Rich formatting
- **Testing**: Comprehensive test suite with mocked tests

### 🗑️ Deprecated

- Moved to `deprecated/`:
  - `preprocess/` (v1.0 Stage1 code)
  - `unet/` (v1.0 Stage2 code)
  - `ultralytics/` (v1.0 YOLO training code)

### 🐛 Fixed

- Corrected `crop_radius` default value from 40 to 25 (matching v1.0)
- Fixed configuration passing in `GeometryEngine` (added `class_id_blank`)
- Added OpenMP conflict handling instructions

### 📦 Dependencies

- `torch>=2.0.0`: PyTorch deep learning framework
- `ultralytics>=8.0.0`: YOLO detection
- `opencv-python>=4.8.0`: Image processing
- `numpy>=1.24.0`: Numerical computation
- `pydantic>=2.0.0`: Data validation
- `PyYAML>=6.0`: Configuration
- `typer>=0.9.0`: CLI framework
- `rich>=13.0.0`: Rich output formatting

**Development**:
- `pytest>=7.0.0`, `pytest-cov>=4.0.0`, `pytest-mock>=3.10.0`
- `black>=23.0.0`, `ruff>=0.1.0`, `mypy>=1.0.0`

---

## [1.0.0] - Previous Version

### Initial Implementation
- YOLO-based chamber detection
- Cross-geometry correction algorithm
- Dual-stream UNet for illumination correction
- ROI weighted loss function
- Full-chip synthesizer

---

**Legend**:
- ✨ Added: New features
- 🔧 Changed: Changes in existing functionality
- 🗑️ Deprecated: Soon-to-be removed features
- 🐛 Fixed: Bug fixes
- 🔒 Security: Vulnerability fixes
