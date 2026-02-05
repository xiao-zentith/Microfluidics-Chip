# Examples

This directory contains example scripts demonstrating how to use the Microfluidics-Chip package.

## Available Examples

### 1. `end_to_end.py` - Complete Pipeline

Demonstrates the full Stage1 + Stage2 processing workflow.

```bash
# Single chip processing
python examples/end_to_end.py 1

# Batch processing
python examples/end_to_end.py 2

# Custom configuration
python examples/end_to_end.py 3

# With Ground Truth
python examples/end_to_end.py 4
```

### 2. `adaptive_detection_demo.py` - 自适应检测示例 (v2.0 新增)

演示粗到精自适应检测 + 拓扑拟合 + 暗腔室判定的完整流程。

```bash
# 基本用法
python examples/adaptive_detection_demo.py --image path/to/image.jpg

# 指定权重和输出目录
python examples/adaptive_detection_demo.py \
    --image path/to/image.jpg \
    --weights weights/yolo/best.pt \
    --output output/demo/

# 使用配置文件
python examples/adaptive_detection_demo.py \
    --image path/to/image.jpg \
    --config configs/adaptive_detection.yaml
```

**输出**：
- `{chip_id}_visualization.png` - 检测结果可视化
- `{chip_id}_detections.json` - 12 个腔室坐标 + 暗腔室标记

## Creating Your Own Scripts

### Basic Template

```python
from pathlib import Path
from microfluidics_chip.core.config import get_default_config
from microfluidics_chip.pipelines.stage1 import run_stage1
from microfluidics_chip.pipelines.stage2 import run_stage2

# Load configuration
config = get_default_config()

# Stage1
stage1_output = run_stage1(
    chip_id="my_chip",
    raw_image_path=Path("data/my_chip.png"),
    gt_image_path=None,
    output_dir=Path("runs/my_experiment/stage1"),
    config=config.stage1
)

# Stage2
stage2_output = run_stage2(
    stage1_run_dir=Path("runs/my_experiment/stage1/my_chip"),
    output_dir=Path("runs/my_experiment/stage2"),
    config=config.stage2
)
```

### With Custom Configuration

```python
from microfluidics_chip.core.config import load_config_from_yaml

config = load_config_from_yaml(Path("configs/my_config.yaml"))
# Then use as above
```

## See Also

- [README.md](../README.md) - Full documentation
- [configs/](../configs/) - Configuration examples
- [scripts/](../scripts/) - Utility scripts
