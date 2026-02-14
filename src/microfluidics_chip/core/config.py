"""
配置管理系统（Pydantic）

支持：
- YAML 配置文件加载
- 环境变量覆盖
- 类型验证
- 默认值管理
"""

from pydantic import BaseModel, Field
from typing import Optional, Tuple, List
from pathlib import Path


# ==================== Stage1 配置 ====================

class YOLOConfig(BaseModel):
    """YOLO 检测器配置"""
    weights_path: str = Field(..., description="YOLO权重文件路径")
    device: str = Field(default="cuda", description="推理设备 (cuda/cpu)")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="检测置信度阈值")
    class_id_blank: int = Field(default=0, description="空白腔类别ID")
    class_id_lit: int = Field(default=1, description="点亮腔类别ID")


class GeometryConfig(BaseModel):
    """几何变换配置"""
    canvas_size: int = Field(default=600, description="理想画布大小（正方形）")
    slice_size: Tuple[int, int] = Field(default=(80, 80), description="切片大小 (H, W)")
    ideal_center_gap: int = Field(default=60, description="第一圈腔室距离中心的距离")
    ideal_chamber_step: int = Field(default=50, description="同一旋臂上腔室的间距")
    crop_radius: int = Field(default=25, description="切片半径（用于裁剪）")
    class_id_blank: int = Field(default=0, description="空白腔类别ID（用于锚点检测）")
    class_id_lit: int = Field(default=1, description="点亮腔类别ID")


# ==================== 自适应检测配置 ====================

class AdaptiveDetectionConfig(BaseModel):
    """自适应粗到精检测配置"""
    # 粗扫描参数
    coarse_imgsz: int = Field(default=640, description="粗扫描分辨率")
    coarse_conf: float = Field(default=0.08, ge=0.0, le=1.0, description="粗扫描置信度阈值")
    
    # 精细扫描参数
    fine_imgsz: int = Field(default=1280, description="精细扫描分辨率")
    fine_conf: float = Field(default=0.3, ge=0.0, le=1.0, description="精细扫描置信度阈值")
    
    # 聚类参数
    cluster_eps: float = Field(default=100.0, description="DBSCAN eps 参数（像素）")
    cluster_min_samples: int = Field(default=3, description="DBSCAN min_samples")
    roi_margin: float = Field(default=1.3, ge=1.0, description="ROI 扩展系数")
    min_roi_size: int = Field(default=200, description="最小 ROI 尺寸")
    
    # 预处理参数
    enable_clahe: bool = Field(default=True, description="是否启用 CLAHE")
    clahe_clip_limit: float = Field(default=2.0, description="CLAHE clip limit")


class TopologyConfig(BaseModel):
    """拓扑拟合配置"""
    # 模板参数
    template_scale: float = Field(default=50.0, description="模板点间距（像素）")
    template_path: Optional[str] = Field(default=None, description="自定义模板文件路径 (JSON)")
    
    # RANSAC 参数
    ransac_iters: int = Field(default=200, description="RANSAC 迭代次数")
    ransac_threshold: float = Field(default=25.0, description="RANSAC 内点阈值（像素）")
    min_inliers: int = Field(default=4, description="最少内点数")
    
    # 可见性判定
    visibility_margin: int = Field(default=10, description="边界安全距离（像素）")
    
    # 暗腔室判定
    brightness_roi_size: int = Field(default=30, description="亮度判定 ROI 尺寸")
    dark_percentile: float = Field(default=25.0, ge=0.0, le=100.0, description="暗腔室判定分位数阈值")
    
    # 回退参数
    fallback_to_affine: bool = Field(default=True, description="Similarity 失败时是否回退到 Affine")


class AdaptiveRuntimeConfig(BaseModel):
    """Stage1 自适应推理运行策略配置"""
    enabled: bool = Field(default=False, description="是否启用自适应粗到精检测流程")
    max_attempts: int = Field(default=3, ge=1, le=6, description="最大重试次数")
    preprocess_sequence: List[str] = Field(
        default_factory=lambda: ["raw", "clahe", "clahe_invert"],
        description="重试预处理序列，可选值: raw/clahe/clahe_invert"
    )

    # 质量闸门
    require_fit_success: bool = Field(default=True, description="是否要求拓扑拟合成功")
    min_detections: int = Field(default=8, ge=0, le=12, description="最少粗细检出数")
    min_inlier_ratio: float = Field(default=0.40, ge=0.0, le=1.0, description="最小RANSAC内点比例")
    max_reprojection_error: float = Field(default=35.0, ge=0.0, description="最大重投影误差（像素）")
    min_cluster_score: float = Field(default=0.20, ge=0.0, le=1.0, description="最小聚类质量得分")
    min_mean_confidence: float = Field(default=0.15, ge=0.0, le=1.0, description="最小平均置信度")
    require_unique_blank: bool = Field(default=True, description="是否要求唯一 blank 锚点")
    require_blank_outermost: bool = Field(default=True, description="是否要求 blank 位于臂最外侧")
    min_arm_monotonicity: float = Field(default=0.75, ge=0.0, le=1.0, description="旋臂内->外距离单调性最小比例")

    # 重试调度
    confidence_decay: float = Field(default=0.85, gt=0.0, le=1.0, description="每次重试置信度衰减系数")
    min_coarse_conf: float = Field(default=0.03, ge=0.0, le=1.0, description="粗检最小置信度")
    min_fine_conf: float = Field(default=0.12, ge=0.0, le=1.0, description="细检最小置信度")
    fine_imgsz_step: int = Field(default=128, ge=0, description="每次重试细检分辨率增量")

    # 兜底策略
    fallback_to_standard: bool = Field(default=True, description="质量闸门失败后是否回退到标准流程")
    force_blank_if_missing: bool = Field(default=True, description="缺少blank类别时是否强制补一个blank锚点")


class Stage1Config(BaseModel):
    """Stage1 完整配置"""
    yolo: YOLOConfig
    geometry: GeometryConfig

    # v2.1: 可选自适应流程配置（默认关闭，兼容旧行为）
    adaptive_detection: Optional[AdaptiveDetectionConfig] = None
    topology: Optional[TopologyConfig] = None
    adaptive_runtime: Optional[AdaptiveRuntimeConfig] = None


# ==================== Stage2 配置 ====================

class UNetModelConfig(BaseModel):
    """UNet 模型配置"""
    model_type: str = Field(default="dual_stream", description="模型类型 (dual_stream/single_stream)")
    in_channels: int = Field(default=3, description="输入通道数")
    out_channels: int = Field(default=3, description="输出通道数")
    features: List[int] = Field(default=[64, 128, 256, 512], description="特征通道数列表")
    device: str = Field(default="cuda", description="推理设备")


class ROILossConfig(BaseModel):
    """ROI 加权损失配置"""
    roi_radius: int = Field(default=20, description="核心反应区半径（像素）")
    edge_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="边缘/背景区域权重")
    lambda_cos: float = Field(default=0.2, ge=0.0, description="余弦相似度损失权重")


class Stage2Config(BaseModel):
    """Stage2 完整配置"""
    model: UNetModelConfig
    loss: ROILossConfig
    weights_path: Optional[str] = Field(default=None, description="UNet权重文件路径")
    
    # 训练数据配置
    reference_chambers: List[int] = Field(
        default=[0, 1, 2], 
        description="基准腔室索引列表（用于双流UNet的参考输入）。例如：[0]表示只用空白腔，[0,1,2]表示使用3个基准腔"
    )
    reference_mode: str = Field(
        default="stack",
        description="基准腔室组合模式: 'stack'=堆叠, 'average'=平均, 'concat'=通道拼接"
    )


# ==================== 全局配置 ====================

class PathsConfig(BaseModel):
    """路径配置"""
    data_dir: str = Field(default="data", description="数据根目录")
    runs_dir: str = Field(default="runs", description="运行输出目录")
    weights_dir: str = Field(default="weights", description="权重文件目录")


class MicrofluidicsConfig(BaseModel):
    """微流控芯片项目完整配置"""
    experiment_name: str = Field(default="microfluidics_default", description="实验名称")
    paths: PathsConfig
    stage1: Stage1Config
    stage2: Stage2Config
    
    class Config:
        # 允许从 dict 构造
        extra = "allow"


# ==================== 配置加载器 ====================

def load_config_from_yaml(yaml_path: Path) -> MicrofluidicsConfig:
    """
    从 YAML 文件加载配置
    
    :param yaml_path: YAML 配置文件路径
    :return: MicrofluidicsConfig 实例
    """
    import yaml
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return MicrofluidicsConfig(**config_dict)


def merge_configs(*configs: MicrofluidicsConfig) -> MicrofluidicsConfig:
    """
    合并多个配置（后者覆盖前者）
    
    优先级示例：
    Priority 1 (Base)   : default.yaml
    Priority 2 (Env)    : local.yaml / remote.yaml
    Priority 3 (Exp)    : experiments/ablation_a.yaml
    
    :param configs: 配置实例列表
    :return: 合并后的配置
    """
    merged_dict = {}
    
    for config in configs:
        merged_dict.update(config.model_dump())
    
    return MicrofluidicsConfig(**merged_dict)


def get_default_config() -> MicrofluidicsConfig:
    """
    获取默认配置（用于测试或快速启动）
    
    注意：实际使用时应从 configs/default.yaml 加载
    """
    return MicrofluidicsConfig(
        experiment_name="default",
        paths=PathsConfig(),
        stage1=Stage1Config(
            yolo=YOLOConfig(
                weights_path="weights/yolo/best.pt",
                device="cuda",
                confidence_threshold=0.5
            ),
            geometry=GeometryConfig(
                canvas_size=600,
                slice_size=(80, 80),
                ideal_center_gap=60,
                ideal_chamber_step=50,
                crop_radius=25
            )
        ),
        stage2=Stage2Config(
            model=UNetModelConfig(
                model_type="dual_stream",
                device="cuda"
            ),
            loss=ROILossConfig(
                roi_radius=20,
                edge_weight=0.1,
                lambda_cos=0.2
            ),
            weights_path="weights/unet/best_model.pth",
            reference_chambers=[0, 1, 2],  # 默认使用3个基准腔室
            reference_mode="stack"
        )
    )
