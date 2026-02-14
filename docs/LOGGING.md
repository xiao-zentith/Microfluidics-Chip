# Microfluidics-Chip 日志系统总结

本文档总结项目中所有日志记录的位置、级别和用途。

---

## 📋 日志系统架构

### 核心模块
**文件**: `src/microfluidics_chip/core/logger.py`

**功能**:
- 统一日志管理
- Rich彩色控制台输出
- 可选文件日志记录
- 多级别日志（DEBUG/INFO/WARNING/ERROR）

**使用方式**:
```python
from microfluidics_chip.core.logger import get_logger

logger = get_logger("module_name")
logger.info("消息")
logger.warning("警告")
logger.error("错误")
```

---

## 📊 日志分布统计

| 模块 | INFO | WARNING | ERROR | 总计 |
|------|------|---------|-------|------|
| **Stage1 Detection** | 13 | 5 | 2 | 20 |
| **Stage2 Correction** | 16 | 3 | 0 | 19 |
| **Pipelines** | 16 | 4 | 2 | 22 |
| **Synthesizer** | 9 | 0 | 2 | 11 |
| **总计** | **54** | **12** | **6** | **72** |

---

## 🔍 详细日志清单

### 1. Stage1 Detection 模块

#### `stage1_detection/detector.py`
```python
logger.info(f"Loading YOLO model from {config.weights_path}...")     # 模型加载开始
logger.info(f"YOLO model loaded on {config.device}")                 # 模型加载完成
logger.warning("No chambers detected in image")                      # 未检测到腔室
```

**用途**: YOLO模型初始化和检测过程跟踪

---

#### `stage1_detection/geometry_engine.py`
```python
logger.info(f"GeometryEngine initialized (canvas={config.canvas_size})")  # 初始化
logger.warning(f"Insufficient detections: {len(detections)} < 12")        # 检测数不足
logger.warning("No blank chamber found. Using fallback arm 0")            # 未找到空白腔
```

**用途**: 几何校正引擎状态和异常情况

---

#### `stage1_detection/inference.py`
```python
# 初始化
logger.info("Initialized new ChamberDetector")
logger.info("Initialized new CrossGeometryEngine")

# 处理流程
logger.info(f"[{chip_id}] Detecting chambers in raw image...")
logger.info(f"[{chip_id}] Detected {len(detections_raw)} chambers")
logger.info(f"[{chip_id}] Processing geometry for raw image...")
logger.info(f"[{chip_id}] Geometry processing complete: {len(chamber_slices)} slices extracted")

# GT处理
logger.info(f"[{chip_id}] Processing GT image with INDEPENDENT engine (P3)")
logger.info(f"[{chip_id}] GT processing complete: {len(gt_slices)} slices")

# 完成
logger.info(f"[{chip_id}] Stage1 inference complete in {processing_time:.2f}s")

# 错误
logger.error(f"[{chip_id}] Insufficient detections: {len(detections_raw)} < 12")
logger.error(f"[{chip_id}] Geometry processing failed")

# 警告
logger.warning(f"[{chip_id}] GT geometry processing failed")
logger.warning(f"[{chip_id}] Insufficient GT detections: {len(detections_gt)}")
```

**用途**: Stage1推理全流程跟踪，带chip_id前缀便于追踪

---

#### `stage1_detection/synthesizer.py`
```python
# 初始化
logger.info("FullChipSynthesizer initialized")

# 合成过程
logger.info(f"Starting synthesis: {len(files)} source images, multiplier={multiplier}")
logger.info(f"Synthesis complete: {len(T)} slices generated")
logger.info(f"Target Shape: {T.shape}")
logger.info(f"Ref Shape:    {R.shape}")
logger.info(f"Label Shape:  {L.shape}")
logger.info(f"Saved to {output_path}")

# 可视化
logger.info(f"Synthesis visualization saved: {save_path}")

# 错误
logger.error(f"Failed to read {clean_full_img_path}")
logger.error(f"Insufficient detections: {len(detections)}")
```

**用途**: 数据合成过程和数据形状验证

---

### 2. Stage2 Correction 模块

#### `stage2_correction/inference.py`
```python
# 初始化
logger.info(f"[{chip_id}] Initializing UNet model...")
logger.info(f"[{chip_id}] Loaded weights from {weights_path}")

# 推理
logger.info(f"[{chip_id}] Correcting {len(chamber_slices)} slices...")
logger.info(f"[{chip_id}] Stage2 inference complete in {processing_time:.2f}s")

# 警告
logger.warning(f"[{chip_id}] Weights file not found: {weights_path}")
```

**用途**: Stage2推理流程跟踪

---

#### `stage2_correction/trainer.py`
```python
# 训练开始
logger.info(f"Starting training for {epochs} epochs")
logger.info(f"Saving to: {save_dir}")

# 每个epoch
logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val PSNR: {v_psnr:.2f} dB | LR: {lr:.2e}")

# 保存最佳模型
logger.info(f"*** Best Model Saved (PSNR: {best_psnr:.2f} dB) ***")

# 训练完成
logger.info("=" * 60)
logger.info("Training complete! Generating training curves...")
logger.info(f"Best PSNR: {best_psnr:.2f} dB at epoch {best_epoch}")
logger.info(f"All results saved to: {save_dir}")
logger.info("=" * 60)

# 可视化
logger.info(f"Visualization saved: {save_path}")
logger.info(f"Training curves saved: {save_path}")
```

**用途**: 训练进度、性能指标和最佳模型跟踪

---

### 3. Pipelines 模块

#### `pipelines/stage1.py`
```python
# 处理开始
logger.info(f"[{chip_id}] Starting Stage1 processing...")

# 调试可视化
logger.info(f"[{chip_id}] Debug visualization saved: debug_detection.png")

# 完成
logger.info(f"[{chip_id}] Stage1 output saved to: {run_dir}")
logger.info(f"[{chip_id}] Files: stage1_metadata.json, aligned.png, chamber_slices.npz")

# 批处理
logger.info(f"Found {len(image_files)} images in {input_dir}")
logger.info(f"Batch processing complete: {success_count} success, {fail_count} failed")

# 错误
logger.error(f"✗ {chip_id} failed: {e}")

# 警告
logger.warning(f"[{chip_id}] Cannot read GT image: {gt_image_path}")
logger.warning(f"No image files found in {input_dir}")
```

**用途**: Stage1业务层处理流程和批处理统计

---

#### `pipelines/stage2.py`
```python
# 处理开始
logger.info(f"[{chip_id}] Starting Stage2 processing...")
logger.info(f"[{chip_id}] Loading Stage1 output from: {stage1_run_dir}")
logger.info(f"[{chip_id}] Loaded {len(chamber_slices)} slices from Stage1")

# 完成
logger.info(f"[{chip_id}] Stage2 output saved to: {run_dir}")
logger.info(f"[{chip_id}] Files: stage2_metadata.json, corrected_slices.npz")

# 批处理
logger.info(f"Found {len(stage1_dirs)} Stage1 output directories")
logger.info("Initializing UNet model for batch processing...")
logger.info(f"Loaded weights from {weights_path}")
logger.info("Model initialized successfully")
logger.info(f"✓ {chip_id} completed ({idx+1}/{len(stage1_dirs)})")
logger.info(f"Batch processing complete: {success_count} success, {fail_count} failed")

# 错误
logger.error(f"✗ {chip_id} failed: {e}")

# 警告
logger.warning(f"No Stage1 output directories found in {stage1_output_dir}")
logger.warning(f"Weights file not found: {weights_path}")
```

**用途**: Stage2业务层处理流程和批处理统计

---

## 🎯 日志级别使用规范

### INFO (`logger.info()`)
**用途**: 正常流程跟踪

**典型场景**:
- 模块初始化完成
- 处理步骤开始/完成
- 数据加载/保存成功
- 批处理进度
- 性能指标（时间、PSNR等）

**示例**:
```python
logger.info(f"[{chip_id}] Detected {len(detections)} chambers")
logger.info(f"Stage1 inference complete in {time:.2f}s")
```

---

### WARNING (`logger.warning()`)
**用途**: 异常情况但不影响继续执行

**典型场景**:
- 检测数量不足但可继续
- 文件未找到但有fallback
- GT处理失败但Raw成功
- 参数使用默认值

**示例**:
```python
logger.warning("No blank chamber found. Using fallback arm 0")
logger.warning(f"Weights file not found: {path}")
```

---

### ERROR (`logger.error()`)
**用途**: 错误导致处理失败

**典型场景**:
- 检测失败（<12个腔室）
- 几何变换失败
- 文件读取失败
- 批处理中单个失败

**示例**:
```python
logger.error(f"[{chip_id}] Insufficient detections: {count} < 12")
logger.error(f"✗ {chip_id} failed: {e}")
```

---

### DEBUG (`logger.debug()`)
**用途**: 详细调试信息（当前项目未使用）

**建议场景**:
- 中间变量值
- 循环迭代详情
- 函数调用追踪

---

## 📁 日志输出位置

### 控制台输出
- **默认**: 使用Rich彩色格式化
- **级别**: INFO及以上
- **格式**: `[时间] 级别 - 消息`

### 文件日志（可选）
- **配置**: `setup_logger(log_file=Path("runs/training.log"))`
- **级别**: DEBUG及以上（记录所有）
- **格式**: `2026-01-28 20:00:00 - microfluidics_chip.module - INFO - 消息`

---

## 🔧 日志使用最佳实践

### 1. 带chip_id前缀
```python
# ✅ 好的做法
logger.info(f"[{chip_id}] Detected {count} chambers")

# ❌ 不好的做法
logger.info("Detected chambers")
```

### 2. 包含关键数值
```python
# ✅ 好的做法
logger.info(f"Stage1 complete in {time:.2f}s, {count} slices")

# ❌ 不好的做法
logger.info("Stage1 complete")
```

### 3. 成功/失败标记
```python
# ✅ 好的做法
logger.info(f"✓ {chip_id} completed")
logger.error(f"✗ {chip_id} failed: {e}")
```

### 4. 分隔符用于重要信息
```python
logger.info("=" * 60)
logger.info("Training complete!")
logger.info("=" * 60)
```

---

## 💡 建议改进

### 1. 添加文件日志（当前未使用）
```python
# 建议在训练脚本中添加
setup_logger(
    name="microfluidics_chip",
    level="INFO",
    log_file=Path("runs/training/training.log")
)
```

### 2. 添加DEBUG级别日志
```python
# 在关键算法中添加
logger.debug(f"Transform matrix: {M}")
logger.debug(f"Detection boxes: {boxes}")
```

### 3. 使用结构化日志
```python
# 可考虑添加JSON格式日志用于分析
import json
logger.info(json.dumps({
    "event": "stage1_complete",
    "chip_id": chip_id,
    "processing_time": time,
    "chamber_count": count
}))
```

### 4. 添加性能统计日志
```python
# 批处理结束时
logger.info(f"Performance: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
```

---

## 📊 日志使用场景总结

| 场景 | 日志位置 | 关键信息 |
|------|---------|---------|
| **模型加载** | detector.py, inference.py | 权重路径、设备 |
| **检测过程** | inference.py | 检测数量、chip_id |
| **几何校正** | geometry_engine.py | 切片数量、异常情况 |
| **推理完成** | inference.py | 处理时间、chip_id |
| **训练过程** | trainer.py | Loss、PSNR、LR |
| **批处理** | stage1.py, stage2.py | 成功/失败统计 |
| **数据合成** | synthesizer.py | 数据形状、保存路径 |

---

**总结**: 项目已有完善的日志系统，覆盖所有关键流程。建议添加文件日志保存和更多DEBUG级别日志用于深度调试。
