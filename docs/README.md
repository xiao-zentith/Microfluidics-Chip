# 文档导航与分工

> 目的：减少重复、避免命令版本漂移，明确每份文档的职责边界。

---

## 命令唯一来源

- `docs/CLI_REFERENCE.md`：CLI 命令主入口（Stage1/Stage2、调试与消融）。
- `python -m microfluidics_chip.pipelines.cli --help`：运行时真实命令定义。

说明：
- `README.md` 只保留快速开始级别命令，不再维护完整 CLI 参数组合。
- 其他 `docs/*.md` 只保留该主题的最小示例，并链接到 `docs/CLI_REFERENCE.md`。

---

## 文档分工

| 文档 | 职责 | 命令策略 |
|------|------|----------|
| `README.md` | 项目总览 + 快速开始 | 最小命令集，指向 CLI 参考 |
| `docs/CLI_REFERENCE.md` | Stage1/Stage2/消融 命令全集 | ✅ 唯一 CLI 命令文档 |
| `docs/DATA_PREPARATION.md` | 数据组织、标注、训练数据准备、评估流程 | 仅训练/数据命令 |
| `docs/ADAPTIVE_DETECTION.md` | Stage1 自适应检测与拓扑后处理策略 | 仅自适应最小闭环命令 |
| `docs/YOLO_OPTIMIZATION.md` | YOLO 调参与精度优化策略 | 仅优化专题命令 |
| `docs/VISUALIZATION.md` | 可视化产物说明与调试建议 | 仅可视化调试命令 |
| `docs/LOGGING.md` | 日志规范与日志点清单 | 以规范为主 |
| `docs/UNET_AUGMENTATION.md` | Stage2 增强策略与接口说明 | 仅 Stage2 增强命令 |
| `docs/TRAINING_IMPROVEMENT.md` | Stage2 训练改进建议与监控 | 仅改进实验命令 |
| `docs/AUGMENTATION_VALIDATION.md` | 增强方法验证报告 | 报告为主 |
| `PROJECT_SUMMARY.md` | 项目阶段性总结（历史快照） | 非实时命令来源 |
| `FINAL_REFACTORING_PLAN.md` | 重构方案文档（历史设计） | 非实时命令来源 |

---

## 推荐阅读路径

1. `README.md`：先跑通安装与最小端到端命令。
2. `docs/CLI_REFERENCE.md`：按场景选择具体 CLI 命令。
3. `docs/DATA_PREPARATION.md`：准备 Stage1/Stage2 训练数据并训练。
4. `docs/ADAPTIVE_DETECTION.md`：理解 Stage1 自适应检测、拓扑回填与质量闸门。
5. 按需阅读专题文档（`docs/YOLO_OPTIMIZATION.md`、`docs/VISUALIZATION.md`、`docs/LOGGING.md`）。
