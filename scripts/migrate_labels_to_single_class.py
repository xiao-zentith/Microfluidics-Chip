#!/usr/bin/env python3
"""
YOLO 标签单类别迁移脚本

将多类别 (chamber_dark=0, chamber_lit=1) 统一为单类别 (chamber=0)。
迁移后，暗腔室判定将通过拓扑拟合 + 亮度分析实现，而非依赖 YOLO 分类。

用法:
    # 预览模式 (不修改文件)
    python scripts/migrate_labels_to_single_class.py --data yolo_v3 --dry-run
    
    # 执行迁移
    python scripts/migrate_labels_to_single_class.py --data yolo_v3
    
    # 指定备份目录
    python scripts/migrate_labels_to_single_class.py --data yolo_v3 --backup-dir backups/
"""

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime

# 添加 src 到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microfluidics_chip.core.logger import setup_logger, get_logger

logger = get_logger("migrate_labels")


def migrate_label_file(label_path: Path, dry_run: bool = False) -> dict:
    """
    将单个标签文件中的所有类别统一为 0
    
    :param label_path: 标签文件路径
    :param dry_run: 是否为预览模式
    :return: 迁移统计 {原类别: 数量}
    """
    stats = {}
    
    content = label_path.read_text(encoding='utf-8').strip()
    if not content:
        return stats
    
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        original_class = int(parts[0])
        stats[original_class] = stats.get(original_class, 0) + 1
        
        # 统一为 class_id=0
        parts[0] = '0'
        new_lines.append(' '.join(parts))
    
    if not dry_run:
        label_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
    
    return stats


def update_data_yaml(yaml_path: Path, dry_run: bool = False) -> None:
    """
    更新 data.yaml 为单类别配置
    
    :param yaml_path: data.yaml 路径
    :param dry_run: 是否为预览模式
    """
    new_content = """# YOLO 数据集配置 (单类别版本)
# 由 migrate_labels_to_single_class.py 生成

train: images/train
val: images/val

nc: 1  # 单类别
names:
  0: chamber  # 所有腔室统一类别

# 暗腔室判定已移至推理阶段 (拓扑拟合 + 亮度分析)
"""
    
    if dry_run:
        logger.info(f"[DRY-RUN] Would update {yaml_path}")
        logger.info(f"New content:\n{new_content}")
    else:
        yaml_path.write_text(new_content, encoding='utf-8')
        logger.info(f"Updated: {yaml_path}")


def update_classes_txt(classes_path: Path, dry_run: bool = False) -> None:
    """
    更新 classes.txt 为单类别
    
    :param classes_path: classes.txt 路径
    :param dry_run: 是否为预览模式
    """
    new_content = "chamber\n"
    
    if dry_run:
        logger.info(f"[DRY-RUN] Would update {classes_path}")
    else:
        classes_path.write_text(new_content, encoding='utf-8')
        logger.info(f"Updated: {classes_path}")


def backup_dataset(dataset_dir: Path, backup_dir: Path) -> Path:
    """
    备份整个数据集
    
    :param dataset_dir: 数据集目录
    :param backup_dir: 备份根目录
    :return: 备份路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{dataset_dir.name}_backup_{timestamp}"
    
    logger.info(f"Creating backup: {backup_path}")
    shutil.copytree(dataset_dir, backup_path)
    logger.info(f"Backup complete: {backup_path}")
    
    return backup_path


def migrate_dataset(
    dataset_dir: Path,
    dry_run: bool = False,
    backup_dir: Path = None,
    skip_backup: bool = False
) -> dict:
    """
    迁移整个数据集
    
    :param dataset_dir: 数据集目录
    :param dry_run: 是否为预览模式
    :param backup_dir: 备份目录
    :param skip_backup: 是否跳过备份
    :return: 迁移统计
    """
    labels_dir = dataset_dir / "labels"
    
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return {}
    
    # 创建备份 (非 dry-run 且未跳过)
    if not dry_run and not skip_backup:
        if backup_dir is None:
            backup_dir = dataset_dir.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_dataset(dataset_dir, backup_dir)
    
    # 收集所有标签文件
    label_files = list(labels_dir.rglob("*.txt"))
    logger.info(f"Found {len(label_files)} label files")
    
    # 迁移统计
    total_stats = {}
    migrated_count = 0
    
    for label_path in label_files:
        stats = migrate_label_file(label_path, dry_run)
        if stats:
            migrated_count += 1
            for cls, count in stats.items():
                total_stats[cls] = total_stats.get(cls, 0) + count
    
    logger.info(f"Migrated {migrated_count} files")
    logger.info(f"Class distribution before migration: {total_stats}")
    
    # 更新配置文件
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        update_data_yaml(data_yaml, dry_run)
    
    classes_txt = dataset_dir / "classes.txt"
    if classes_txt.exists():
        update_classes_txt(classes_txt, dry_run)
    
    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate YOLO labels to single class",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="数据集名称 (在 data/stage1_detection/ 下的目录名)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不实际修改文件"
    )
    
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="备份目录 (默认: data/stage1_detection/backups/)"
    )
    
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="跳过备份 (不推荐)"
    )
    
    args = parser.parse_args()
    
    setup_logger(level="INFO")
    
    # 构建数据集路径
    dataset_dir = PROJECT_ROOT / "data" / "stage1_detection" / args.data
    
    if not dataset_dir.exists():
        logger.error(f"Dataset not found: {dataset_dir}")
        available = list((PROJECT_ROOT / "data" / "stage1_detection").iterdir())
        logger.info(f"Available datasets: {[d.name for d in available if d.is_dir()]}")
        return 1
    
    if args.dry_run:
        logger.info("=" * 50)
        logger.info("DRY-RUN MODE - No files will be modified")
        logger.info("=" * 50)
    
    logger.info(f"Dataset: {dataset_dir}")
    
    stats = migrate_dataset(
        dataset_dir=dataset_dir,
        dry_run=args.dry_run,
        backup_dir=args.backup_dir,
        skip_backup=args.skip_backup
    )
    
    if args.dry_run:
        logger.info("=" * 50)
        logger.info("DRY-RUN complete. Run without --dry-run to apply changes.")
        logger.info("=" * 50)
    else:
        logger.info("=" * 50)
        logger.info("Migration complete!")
        logger.info(f"New class structure: nc=1, names={{0: 'chamber'}}")
        logger.info("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
