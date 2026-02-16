# Stage1 BLANK: Chromaticity Mode (v2)

## 目标
- 不改 topology / geometry 主流程。
- 在 `stage1-post` 新增 `blank_mode=chromaticity`。
- 用颜色差分特征（donut-ring）替代“亮度最小”规则，降低光照不均、反光导致的 BLANK 误判。

## 方法概要
1. 每个 chamber 在 canonical 图上取两层 ROI：
   - `donut`: `[0.25r, 0.55r]`
   - `ring`: `[1.10r, 1.40r]`
2. 计算 donut/ring 的 robust median（RGB + HSV-S）。
3. 差分特征：
   - `df_r` (chromaticity r 差)
   - `df_rg` (R/G ratio 差)
   - `df_re` (red excess 差)
   - `df_s` (saturation 差)
4. 对 `df_rg`、`df_re` 做鲁棒裁剪（分位数 + 最小裁剪阈值）。
5. `reaction_score`（越小越像参考臂/空白）：
   - `w1*abs(df_r) + w2*abs(df_rg) + w3*max(df_re,0) + w4*max(df_s,0)`
6. `reference_arm = argmin(median(score of 3 chambers in arm))`
7. `blank = argmin(score in reference_arm)`
8. margin gate：
   - `arm_margin = second_best_arm - best_arm`
   - `blank_margin = second_best_chamber - best_chamber`
   - 若任一 margin 低于阈值，标记 `blank_unresolved`。

## 新增配置项
文件：`configs/default.yaml` / `configs/two_class.yaml`

- `stage1.topology.blank_mode`: `brightness|color|chromaticity`
- `stage1.topology.blank_chroma_w1`
- `stage1.topology.blank_chroma_w2`
- `stage1.topology.blank_chroma_w3`
- `stage1.topology.blank_chroma_w4`
- `stage1.topology.blank_chroma_arm_margin_thr`
- `stage1.topology.blank_chroma_blank_margin_thr`
- `stage1.topology.blank_chroma_clip_quantile`
- `stage1.topology.blank_chroma_clip_rg_min`
- `stage1.topology.blank_chroma_clip_re_min`

## Debug 输出
每次 `stage1-post` 都会输出：
- `debug_stage1_topology.png`
- `debug_stage1_topology.json`
- `debug_blank_features.json`（新增）

`debug_blank_features.json` 包含：
- 每个 chamber 的 donut/ring 统计
- `df_*` 特征
- `reaction_score`
- `arm_scores`、`reference_arm`
- `blank_id_pred`、`arm_margin`、`blank_margin`
- `score_method`、`score_weights`

## 批量对比脚本
新增脚本：`scripts/validate_blank_modes_batch.py`

用途：同一批样本上对比 `brightness` vs `chromaticity`。

输出：
- `summary.csv`（每样本每模式一行）
- `compare_summary.csv`（每样本 old/new 对照）
- `key_cases/`（blank 改变、margin 低、或 unresolved 的样本）

## 可运行命令
使用 detection json 根目录（推荐）：

```bash
python scripts/validate_blank_modes_batch.py \
  --input-dir data/experiments/stage1_yolo_adaptive \
  --output-dir data/experiments/blank_mode_compare \
  --config configs/default.yaml \
  --template configs/templates/pinwheel_v3_centered.json \
  --max-samples 20 \
  --min-topology-detections 8
```

使用 stage1-post 结果根目录（脚本会自动转换 `debug_stage1_topology.json` 为可重跑输入）：

```bash
python scripts/validate_blank_modes_batch.py \
  --input-dir data/experiments/stage1_post_debug \
  --output-dir data/experiments/blank_mode_compare_from_post \
  --config configs/default.yaml \
  --template configs/templates/pinwheel_v3_centered.json \
  --max-samples 20
```
