# CLI 命令参考（唯一入口）

> 本文档维护 `microfluidics_chip.pipelines.cli` 的常用命令示例与参数组合。  
> 运行时真实定义请以 `python -m microfluidics_chip.pipelines.cli --help` 为准。

---

## Stage1

### 1） 标准 Stage1（单图）

```bash
python -m microfluidics_chip.pipelines.cli stage1 IMAGE_PATH -o OUTPUT_DIR
```

常见组合：

```bash
# 带 GT
python -m microfluidics_chip.pipelines.cli stage1 data/chip001.png --gt data/chip001_gt.png -o data/experiments/stage1

# 调试输出
python -m microfluidics_chip.pipelines.cli stage1 data/chip001.png -o data/experiments/debug --save-slices --save-debug --adaptive

# 关闭自适应
python -m microfluidics_chip.pipelines.cli stage1 data/chip001.png -o data/experiments/stage1 --no-adaptive
```

### 2) 标准 Stage1（批量）

```bash
python -m microfluidics_chip.pipelines.cli stage1-batch data/images -o data/experiments/stage1_batch --adaptive
```

### 3) 消融：仅普通 YOLO（无后处理）

```bash
python -m microfluidics_chip.pipelines.cli stage1-yolo data/chip001.png -o data/experiments/stage1_yolo
python -m microfluidics_chip.pipelines.cli stage1-yolo-batch data/images -o data/experiments/stage1_yolo
```

### 4) 消融：仅两阶段 YOLO（无后处理）

```bash
python -m microfluidics_chip.pipelines.cli stage1-yolo-adaptive data/chip001.png -o data/experiments/stage1_yolo_adaptive
python -m microfluidics_chip.pipelines.cli stage1-yolo-adaptive-batch data/images -o data/experiments/stage1_yolo_adaptive
```

### 5) 消融：仅后处理（拓扑回填 + BLANK 末端判定）

```bash
python -m microfluidics_chip.pipelines.cli stage1-post \
  data/experiments/stage1_yolo_adaptive/chip001/adaptive_yolo_raw_detections.json \
  -o data/experiments/stage1_post
```

可选参数：

```bash
# 调整最小拓扑拟合点数阈值（默认跟随配置 min_detections）
python -m microfluidics_chip.pipelines.cli stage1-post <detections_json> -o data/experiments/stage1_post --min-topology-detections 8

# 关闭 fallback 重检
python -m microfluidics_chip.pipelines.cli stage1-post <detections_json> -o data/experiments/stage1_post --no-fallback-detection
```

批量后处理：

```bash
python -m microfluidics_chip.pipelines.cli stage1-post-batch \
  data/experiments/stage1_yolo_adaptive \
  -o data/experiments/stage1_post \
  --json-name adaptive_yolo_raw_detections.json
```

### 6) BLANK 模式对比（brightness vs chromaticity）

```bash
python scripts/validate_blank_modes_batch.py \
  --input-dir data/experiments/stage1_yolo_adaptive \
  --output-dir data/experiments/blank_mode_compare \
  --config configs/default.yaml \
  --template configs/templates/pinwheel_v3_centered.json \
  --max-samples 20
```

输出：
- `summary.csv`：每样本每模式一行（含 `blank_mode/reference_arm_pred/blank_id_pred/arm_margin/blank_margin/blank_unresolved`）
- `compare_summary.csv`：old/new 对比
- `key_cases/`：发生变化或低 margin 的样本

---

## Stage2

### 1) Stage2（单图）

```bash
python -m microfluidics_chip.pipelines.cli stage2 data/experiments/stage1/chip001 -o data/experiments/stage2
```

### 2) Stage2（批量）

```bash
python -m microfluidics_chip.pipelines.cli stage2-batch data/experiments/stage1 -o data/experiments/stage2
```

---

## 典型链路

```bash
# A. 端到端（推荐）
python -m microfluidics_chip.pipelines.cli stage1 data/chip001.png -o data/experiments/stage1 --adaptive
python -m microfluidics_chip.pipelines.cli stage2 data/experiments/stage1/chip001 -o data/experiments/stage2
```

```bash
# B. 消融链路（检测与后处理解耦）
python -m microfluidics_chip.pipelines.cli stage1-yolo-adaptive data/chip001.png -o data/experiments/stage1_yolo_adaptive
python -m microfluidics_chip.pipelines.cli stage1-post data/experiments/stage1_yolo_adaptive/chip001/adaptive_yolo_raw_detections.json -o data/experiments/stage1_post
```
