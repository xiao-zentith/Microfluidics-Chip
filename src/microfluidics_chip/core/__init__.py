"""
Core module: 核心公共模块

包含：
- types: 数据类型定义（Result/Output双层）
- io: 统一IO管理（ResultSaver）
- config: 配置管理（Pydantic）
- logger: 日志系统
"""

from .types import (
    ChamberDetection,
    TransformParams,
    Stage1Result,
    Stage1Output,
    Stage2Result,
    Stage2Output,
    stage1_result_to_output,
    stage2_result_to_output,
)

from .io import (
    ResultSaver,
    save_stage1_result,
    load_stage1_output,
    save_stage2_result,
    load_stage2_output,
)

__all__ = [
    # Types
    "ChamberDetection",
    "TransformParams",
    "Stage1Result",
    "Stage1Output",
    "Stage2Result",
    "Stage2Output",
    "stage1_result_to_output",
    "stage2_result_to_output",
    # IO
    "ResultSaver",
    "save_stage1_result",
    "load_stage1_output",
    "save_stage2_result",
    "load_stage2_output",
]
