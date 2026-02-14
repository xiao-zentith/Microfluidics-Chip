"""
数据集图像标准化重命名脚本

功能：
1. 智能识别GT图像和Dirty图像
2. 标准化命名为 gt.png 和 dirty_01.png, dirty_02.png, ...
3. 预览模式（dry-run）避免误操作
4. 自动备份原始文件名映射

用法：
    # 预览重命名（不执行）
    python scripts/rename_dataset.py dataset/chip001 --dry-run
    
    # 执行重命名
    python scripts/rename_dataset.py dataset/chip001
    
    # 手动指定GT图像
    python scripts/rename_dataset.py dataset/chip001 --gt-image IMG_1234.png
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional


def find_images(directory: Path) -> List[Path]:
    """查找目录中的所有图像文件"""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    images_set = set()  # 使用集合去重
    
    for ext in extensions:
        for img in directory.glob(ext):
            # 使用绝对路径去重（Windows下大小写不敏感）
            images_set.add(img.resolve())
    
    return sorted(list(images_set))


def identify_gt_image(images: List[Path]) -> Optional[Path]:
    """智能识别GT图像
    
    优先级：
    1. 文件名包含 'gt', 'GT', 'clean', 'reference'
    2. 文件名包含 'ideal', 'standard', 'baseline'
    3. 文件大小最大（通常GT图像质量最好，压缩率低）
    """
    if not images:
        return None
    
    # 优先级1：关键词匹配
    gt_keywords = ['gt', 'GT', 'clean', 'reference', 'ideal', 'standard', 'baseline', 'blank']
    for img in images:
        name_lower = img.stem.lower()
        for keyword in gt_keywords:
            if keyword.lower() in name_lower:
                print(f"  [+] 通过关键词识别GT: '{img.name}' (包含 '{keyword}')")
                return img
    
    # 优先级2：如果只有一张图像，询问用户
    if len(images) == 1:
        print(f"  [!] 只有一张图像: {images[0].name}")
        return images[0]
    
    # 优先级3：文件大小（通常GT质量最好）
    largest = max(images, key=lambda p: p.stat().st_size)
    print(f"  [?] 无法通过文件名识别GT，选择文件最大的: '{largest.name}' ({largest.stat().st_size / 1024:.1f} KB)")
    print(f"      如不正确，请使用 --gt-image 手动指定")
    return largest


def preview_rename_plan(
    images: List[Path],
    gt_image: Path,
    output_dir: Optional[Path] = None
) -> List[Tuple[Path, Path]]:
    """生成重命名计划
    
    返回: [(old_path, new_path), ...]
    """
    if output_dir is None:
        output_dir = images[0].parent
    
    rename_plan = []
    dirty_counter = 1
    
    for img in images:
        if img == gt_image:
            # GT图像重命名为 gt.png
            new_name = f"gt{img.suffix}"
            new_path = output_dir / new_name
        else:
            # Dirty图像重命名为 dirty_01.png, dirty_02.png, ...
            new_name = f"dirty_{dirty_counter:02d}{img.suffix}"
            new_path = output_dir / new_name
            dirty_counter += 1
        
        rename_plan.append((img, new_path))
    
    return rename_plan


def execute_rename(rename_plan: List[Tuple[Path, Path]], backup: bool = True):
    """执行重命名
    
    :param rename_plan: [(old_path, new_path), ...]
    :param backup: 是否备份文件名映射
    """
    if not rename_plan:
        print("没有需要重命名的文件")
        return
    
    # 创建备份映射
    if backup:
        backup_dir = rename_plan[0][0].parent / ".rename_backup"
        backup_dir.mkdir(exist_ok=True)
        
        backup_mapping = {
            "timestamp": datetime.now().isoformat(),
            "mappings": [
                {
                    "old": str(old.name),
                    "new": str(new.name),
                    "old_size": old.stat().st_size
                }
                for old, new in rename_plan
            ]
        }
        
        backup_file = backup_dir / f"rename_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"[+] 备份映射已保存: {backup_file}")
    
    # 执行重命名（使用临时名避免冲突）
    temp_mappings = []
    
    for i, (old_path, new_path) in enumerate(rename_plan):
        # 第一步：重命名为临时名
        temp_path = old_path.parent / f".tmp_rename_{i}{old_path.suffix}"
        old_path.rename(temp_path)
        temp_mappings.append((temp_path, new_path))
    
    # 第二步：临时名重命名为最终名
    for temp_path, new_path in temp_mappings:
        temp_path.rename(new_path)
        print(f"  [+] {temp_path.name} -> {new_path.name}")


def restore_from_backup(backup_file: Path, directory: Path):
    """从备份恢复原始文件名"""
    with open(backup_file, 'r', encoding='utf-8') as f:
        backup_data = json.load(f)
    
    print(f"从备份恢复: {backup_file}")
    print(f"备份时间: {backup_data['timestamp']}")
    
    for mapping in backup_data['mappings']:
        old_name = mapping['old']
        new_name = mapping['new']
        
        current_path = directory / new_name
        original_path = directory / old_name
        
        if current_path.exists():
            current_path.rename(original_path)
            print(f"  [+] {new_name} -> {old_name}")
        else:
            print(f"  [!] 文件不存在: {new_name}")


def main():
    parser = argparse.ArgumentParser(
        description="标准化重命名数据集图像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "directory",
        type=Path,
        help="包含图像的目录"
    )
    
    parser.add_argument(
        "--gt-image",
        type=str,
        default=None,
        help="手动指定GT图像文件名（如果自动识别错误）"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不实际执行重命名"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不创建备份映射（不推荐）"
    )
    
    parser.add_argument(
        "--restore",
        type=Path,
        default=None,
        help="从备份文件恢复原始文件名"
    )
    
    args = parser.parse_args()
    
    # 恢复模式
    if args.restore:
        restore_from_backup(args.restore, args.directory)
        return 0
    
    # 检查目录
    if not args.directory.exists():
        print(f"[!] 目录不存在: {args.directory}")
        return 1
    
    # 查找图像
    print(f"[*] 扫描目录: {args.directory}")
    images = find_images(args.directory)
    
    if not images:
        print(f"[!] 未找到图像文件")
        return 1
    
    print(f"[+] 找到 {len(images)} 张图像:")
    for img in images:
        size_kb = img.stat().st_size / 1024
        print(f"  - {img.name} ({size_kb:.1f} KB)")
    
    # 识别GT图像
    print("\n[*] 识别GT图像...")
    if args.gt_image:
        gt_image = args.directory / args.gt_image
        if not gt_image.exists():
            print(f"[!] 指定的GT图像不存在: {args.gt_image}")
            return 1
        print(f"  [+] 使用用户指定的GT: {gt_image.name}")
    else:
        gt_image = identify_gt_image(images)
        if not gt_image:
            print(f"[!] 无法识别GT图像，请使用 --gt-image 手动指定")
            return 1
    
    # 生成重命名计划
    print("\n[*] 生成重命名计划...")
    rename_plan = preview_rename_plan(images, gt_image)
    
    # 显示计划
    print("\n" + "=" * 70)
    print("重命名计划:")
    print("=" * 70)
    for old, new in rename_plan:
        marker = "[GT]" if old == gt_image else "[  ]"
        print(f"{marker} {old.name:40s} -> {new.name}")
    print("=" * 70)
    
    # 检查冲突
    conflicts = []
    for _, new_path in rename_plan:
        if new_path.exists() and new_path not in [old for old, _ in rename_plan]:
            conflicts.append(new_path)
    
    if conflicts:
        print("\n[!] 警告：以下文件将被覆盖:")
        for conflict in conflicts:
            print(f"  - {conflict.name}")
        
        if not args.dry_run:
            response = input("\n是否继续？(yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("已取消")
                return 0
    
    # 执行或预览
    if args.dry_run:
        print("\n[DRY-RUN] 预览模式（未实际执行）")
        print("如确认无误，移除 --dry-run 参数重新运行")
    else:
        print("\n[+] 开始重命名...")
        execute_rename(rename_plan, backup=not args.no_backup)
        print("\n[+] 重命名完成！")
        
        # 验证结果
        print("\n[*] 验证结果:")
        gt_files = list(args.directory.glob("gt.*"))
        dirty_files = sorted(args.directory.glob("dirty_*.png")) + sorted(args.directory.glob("dirty_*.jpg"))
        
        print(f"  GT图像:     {len(gt_files)} 张")
        print(f"  Dirty图像:  {len(dirty_files)} 张")
        
        if len(gt_files) == 1 and len(dirty_files) > 0:
            print("\n  [+] 目录结构正确！可以开始训练。")
        else:
            print("\n  [!] 结果异常，请检查")
    
    return 0


if __name__ == "__main__":
    exit(main())
