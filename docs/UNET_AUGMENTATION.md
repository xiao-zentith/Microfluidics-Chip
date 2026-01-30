# UNet 数据增强方案 (v1.2)

> **版本**: v1.2  
> **原则**: 物理一致性 · 离线光学+在线几何

---

## 增强架构

```
┌─────────────────────────────────────────────────────────────┐
│              v1.2 两阶段增强架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [离线阶段] prepare_training_data.py                        │
│  ├── ISP退化链 (全图级别，切片前)                            │
│  │   ├── 光照场 (渐晕 + 方向性光源)                          │
│  │   ├── 白平衡漂移                                         │
│  │   ├── 曝光增益                                           │
│  │   ├── Gamma校正                                          │
│  │   └── Shot Noise (光子计数模型)                          │
│  └── 保证: signal与ref在同一光照场                          │
│                                                             │
│  [在线阶段] MicrofluidicDataset.__getitem__()               │
│  ├── 几何增强 (三元组同步)                                   │
│  │   ├── 水平/垂直翻转 (p=0.5)                              │
│  │   └── 90°旋转 (可配置)                                   │
│  └── 禁止: 任何颜色/光照调整                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 使用方法

### 1. 数据准备（离线增强）

```bash
# 不使用增强
python scripts/prepare_training_data.py data/raw/microfluidics_v1/training -o data/processed/microfluidics_v1/training.npz

# 使用5倍ISP增强
python scripts/prepare_training_data.py data/raw/microfluidics_v1/training -o data/processed/microfluidics_v1/training.npz \
    --augment --aug-multiplier 5
```

### 2. 训练（在线几何增强）

```bash
python scripts/train_stage2.py data/processed/microfluidics_v1/training.npz -o data/experiments/baseline \
    --augment  # 启用在线几何增强
```

---

## ISP退化链公式

$$I_{aug} = \text{Gamma}\big(\text{Exposure}\big(\text{WhiteBal}(I_{raw} \cdot M_{illum})\big)\big) + N_{shot}$$

### 参数

| 组件 | 参数范围 | 物理意义 |
|------|---------|---------|
| 光照场 $M_{illum}$ | strength=0.3 | 渐晕+方向光 |
| 白平衡 | r,b_gain∈[0.9,1.1] | LED色温 |
| 曝光 | gain∈[0.85,1.15] | AE不稳定 |
| Gamma | γ∈[0.8,1.2] | 色调映射 |
| Shot Noise | $N_{peak}$∈[30,100] | 光子计数 |

### Shot Noise模型

$$I_{noisy} = \frac{\text{Poisson}(I_{clean} \cdot N_{peak})}{N_{peak}}$$

---

## API

### 在线增强

```python
from microfluidics_chip.stage2_correction.augmentations import (
    GeometricAugmentation,
    get_train_augmentation
)

# 获取增强器
aug = get_train_augmentation(rotate90=True)

# 应用 (三元组同步)
signal, ref, gt = aug(signal, ref, gt)
```

### 离线增强

```python
from microfluidics_chip.stage2_correction.augmentations import (
    apply_isp_degradation,
    apply_shot_noise
)

# 全图ISP退化
degraded = apply_isp_degradation(image_uint8)

# 仅Shot Noise
noisy = apply_shot_noise(image_float32, peak_photon_count=50)
```

---

## 变更日志

### v1.2 (当前)
- 在线增强简化为仅几何变换
- 光学增强移至离线阶段
- 使用光子计数Shot Noise模型
- 废弃 `MicrofluidicAugmentation` 类

### v1.1
- 添加混合噪声模型
- 自适应增强策略

### v1.0
- 初始版本，基于synthesizer_chip.py
