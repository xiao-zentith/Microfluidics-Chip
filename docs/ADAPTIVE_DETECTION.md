# Stage1 检测精度提升方案

> **目标**: 确保在复杂光照、尺度变化条件下稳定检测 12 个腔室，并正确识别唯一的暗腔室

---

## 🎯 整体架构

```
┌─────────────────────────── 数据层 ───────────────────────────┐
│  单类别标签  →  分层增强 (mild/medium/extreme)  →  增强数据集  │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────── 推理层 ───────────────────────────┐
│  原图 → 粗扫描 → DBSCAN聚类 → ROI提取 → 精扫描 → 拓扑拟合 → 12腔室 │
└───────────────────────────────────────────────────────────────┘
```

---

## 📊 第1层：数据与标签策略

### 1.1 单类别标签迁移

将原有的 `chamber_dark` / `chamber_lit` 合并为统一的 `chamber` 类别，暗腔室判定移至推理层。

```bash
python scripts/migrate_labels_to_single_class.py \
    --root data/stage1_detection/yolo_v3 \
    --dry-run  # 先预览
```

### 1.2 分层离线增强 (v2.2)

| 档位 | 比例 | 光照强度 | 光子计数 | 适用场景 |
|------|------|----------|----------|----------|
| mild | 70% | 0.1-0.3 | 60-100 | 正常条件 |
| medium | 25% | 0.3-0.5 | 40-60 | 轻微退化 |
| extreme | 5% | 0.5-0.8 | 20-40 | 极端条件 |

**关键改进**：
- CLAHE 默认在退化**之后**执行，避免二次噪声放大
- 每副本独立预处理，增加多样性
- 质量控制机制，防止 extreme 产生不可用样本

```bash
# 推荐用法 (默认: multiplier=3, prob=0.7)
python scripts/augment_yolo_dataset.py \
    --input data/stage1_detection/yolo_v3/images/train

# 自定义配置
python scripts/augment_yolo_dataset.py \
    --input data/stage1_detection/yolo_v3/images/train \
    --multiplier 2 --prob 0.8 \
    --clahe-position before_degradation \
    --invert-prob 0.05 --verbose
```

---

## 🔍 第2层：自适应推理 Pipeline

### 2.1 粗到精检测 (AdaptiveDetector)

| 阶段 | 置信度 | 分辨率 | 目的 |
|------|--------|--------|------|
| 粗扫描 | 0.15 | 原图 | 找到大致区域 |
| ROI 提取 | - | DBSCAN 聚类 | 自动定位芯片 |
| 精扫描 | 0.4 | ROI + CLAHE | 精确检测 |

### 2.2 拓扑约束拟合 (TopologyFitter)

**生化背景与检测原理**：
- **参考臂 (Reference Arm)**：
    - **暗腔室 (Dark Chamber)**：位于最外侧。**只含待测液**（无酶、无纳米颗粒）。因此呈现**最低亮度**。
    - **其他腔室**：含待测液 + 酶（无纳米颗粒）。亮度可能不低，因此**不能**作为判定依据。
- **反应臂 (Reaction Arms)**：
    - 3条臂分别检测葡萄糖/尿酸/胆固醇。
    - 含待测液 + 酶 + **纳米颗粒**。
    - 纳米颗粒通常引入显色/荧光/浊度，使得其最外侧腔室显著亮于暗腔室。

**判定核心逻辑**：
利用 **"有无纳米颗粒"** 产生的物理外观差异，对比 4 个臂的**最外侧**腔室。
- **目标**：寻找唯一的“纯待测液”腔室。
- **策略**：忽略臂上中间腔室（排除干扰），只通过最外侧的“暗腔室 candidate”与其余 3 个“反应腔室”对比。

**十字模板结构**（硬编码先验）：
```
          0
          1
          2 ← 最外侧 (候选暗腔室)
  9 10 11     3 4 5
          6
          7
          8 ← 最外侧 (候选暗腔室)
```

| 特性 | 值 |
|------|-----|
| 臂数 | 4 条 |
| 每臂腔室数 | 3 个 |
| 中心腔室 | **无** |
| 暗腔室数量 | **唯一 1 个** |
| 暗腔室位置 | 臂最外侧 `[2, 5, 8, 11]` |

**暗腔室判定逻辑 (v1.4 Safety First)**：
1. **范围锁定**：只在 4 个最外侧位置寻找。
2. **强制竞择**：选出其中亮度最低的 Candidate。
3. **双重保险验证**：
    - **强证据** (`Ratio < 0.80`)：✅ 差异显著，直接确信。
    - **弱证据** (`Ratio < 0.90` AND `Value < 80`)：✅ 差异中等但位置确实很黑，确信。
    - **证据不足**：❌ 返回空。**宁可漏判 (Fail Fast)，不可误判 (防止 Stage 2 基准错误)**。

**RANSAC 拟合**：
- 使用 Similarity Transform（旋转 + 缩放 + 平移）
- 最少需要 4 个内点
- 自动回填漏检腔室坐标

---

## 🛠️ 关键文件

| 类型 | 文件 | 功能 |
|------|------|------|
| 脚本 | `scripts/migrate_labels_to_single_class.py` | 标签迁移 |
| 脚本 | `scripts/augment_yolo_dataset.py` (v2.2) | 离线增强 |
| 核心 | `stage1_detection/preprocess.py` | 统一预处理 |
| 核心 | `stage1_detection/adaptive_detector.py` | 粗到精检测 |
| 核心 | `stage1_detection/topology_fitter.py` | RANSAC 拟合 |
| 入口 | `stage1_detection/inference.py` | `infer_stage1_adaptive()` |
| 配置 | `configs/adaptive_detection.yaml` | 参数模板 |
| 示例 | `examples/adaptive_detection_demo.py` | 端到端演示 |

---

## 🚀 完整使用流程

```bash
# Step 1: 标签迁移
python scripts/migrate_labels_to_single_class.py --root data/stage1_detection/yolo_v3

# Step 2: 数据增强
python scripts/augment_yolo_dataset.py --input data/stage1_detection/yolo_v3/images/train

# Step 3: 重新训练 YOLO
python scripts/train_yolo.py --data data/stage1_detection/yolo_v3_augmented/data.yaml

# Step 4: 使用自适应推理
python examples/adaptive_detection_demo.py --image test.jpg --weights weights/best.pt
```

---

## 📈 预期效果

| 指标 | 原方案 | 新方案 |
|------|--------|--------|
| 检测召回率 | ~80-90% | ~95%+ (拓扑回填) |
| 暗腔室识别 | 依赖检测 | 拓扑约束 + 亮度判定 |
| 尺度鲁棒性 | 单尺度 | 粗到精自适应 |
| 光照鲁棒性 | 一般 | CLAHE + 分层增强 |

---

## 🔗 相关文档

- [YOLO_OPTIMIZATION.md](./YOLO_OPTIMIZATION.md) - YOLO 调参技巧
- [AUGMENTATION_VALIDATION.md](./AUGMENTATION_VALIDATION.md) - 增强验证方法
- [DATA_PREPARATION.md](./DATA_PREPARATION.md) - 数据准备流程
