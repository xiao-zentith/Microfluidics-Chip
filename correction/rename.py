import os
import shutil
import re
from pathlib import Path

# --- 1. 配置 (您需要修改的部分) ---

# (!!!) 关键：这是您“未排序”的源数据文件夹
# (即包含 'chip1', 'chip2'... 文件夹的那个文件夹)
SOURCE_DIR = Path("/home/asus515/PycharmProjects/YOLO_v11/dataset/correction_data/11.12")

# (!!!) 关键：这是您“已排序”的目标文件夹
# (这个路径必须与 `run_correction_pipeline.py` (run_correction_pipeline.py) 中的 `DATASET_DIR` 匹配)
DEST_DIR = Path("/home/asus515/PycharmProjects/YOLO_v11/dataset/correction_data/val")

# (!!!) 命名约定 (与 `run_correction_pipeline.py` (run_correction_pipeline.py) 匹配)
BASE_ID_PREFIX = "S" 
CLEAN_SUFFIX = "_clean"
DIRTY_SUFFIX = "_dirty"

# --- 2. 自动创建目标文件夹 ---
# (我们假设您只用 'val' 集来做这个校正测试)
DEST_DIRTY = DEST_DIR / "images_dirty" / "val"
DEST_CLEAN = DEST_DIR / "images_clean" / "val"

print(f"正在创建目标文件夹 (如果不存在)...")
DEST_DIRTY.mkdir(parents=True, exist_ok=True)
DEST_CLEAN.mkdir(parents=True, exist_ok=True)

# --- 3. 辅助函数：按数字排序 ---
def sort_key_natural(folder_name):
    """
    一个“自然排序”键, 确保 'chip10' 在 'chip9' 之后。
    """
    # 从 'chip1', 'chip10' 中提取数字
    numbers = re.findall(r'\d+', folder_name.name)
    if numbers:
        return int(numbers[0])
    return 0 # 如果没有数字，排在最前

# --- 4. 主处理循环 ---
def process_folders():
    print(f"--- 正在扫描源文件夹: {SOURCE_DIR} ---")
    
    if not SOURCE_DIR.exists():
        print(f"*** 错误: 找不到源文件夹 '{SOURCE_DIR}'。")
        print("    请先创建它, 并放入 'chip1', 'chip2'... 文件夹。")
        return

    # 1. 获取所有样本文件夹 (例如 "chip1", "chip2" ...)
    try:
        sample_folders = sorted(
            [f for f in SOURCE_DIR.iterdir() if f.is_dir()],
            key=sort_key_natural # (!!!) 使用自然数字排序
        )
    except Exception as e:
        print(f"排序时出错: {e}, 将使用字母排序。")
        sample_folders = sorted([f for f in SOURCE_DIR.iterdir() if f.is_dir()])

    if not sample_folders:
        print(f"*** 错误: 在 {SOURCE_DIR} 中没有找到任何 'chipX' 文件夹。")
        return

    print(f"找到了 {len(sample_folders)} 个样本文件夹 (chip1 ... chip7)。")

    total_clean_copied = 0
    total_dirty_copied = 0

    # 2. 遍历每个样本文件夹
    for i, sample_dir in enumerate(sample_folders):
        
        # (!!!) 自动生成 'S001', 'S002' ...
        base_id = f"{BASE_ID_PREFIX}{i+1:03d}" 
        print(f"\n--- 正在处理文件夹 '{sample_dir.name}' (指定ID: {base_id}) ---")
        
        clean_folder = sample_dir / "clean"
        dirty_folder = sample_dir / "dirty"

        # 3. 查找并处理“净图” (clean)
        if not clean_folder.exists():
            print(f"  [警告] 找不到 'clean' 文件夹。跳过此芯片。")
            continue
            
        clean_images = list(clean_folder.glob("*.jpg")) + list(clean_folder.glob("*.png")) + list(clean_folder.glob("*.jpeg"))
        
        if not clean_images:
            print(f"  [警告] 在 'clean' 文件夹中找不到净图。跳过此芯片。")
            continue
        
        if len(clean_images) > 1:
            print(f"  [警告] 'clean' 文件夹中有多张图，将只使用第一张: {clean_images[0].name}")
        
        clean_file_path = clean_images[0]
        clean_ext = clean_file_path.suffix
        
        # 4. 移动并重命名“净图”
        new_clean_name = f"{base_id}{CLEAN_SUFFIX}{clean_ext}"
        dest_clean_path = DEST_CLEAN / new_clean_name
        shutil.copy(str(clean_file_path), str(dest_clean_path)) # 使用 copy 更安全
        print(f"  (答案) -> {new_clean_name} (已保存)")
        total_clean_copied += 1

        # 5. 查找并处理所有“脏图” (dirty)
        if not dirty_folder.exists():
            print(f"  [警告] 找不到 'dirty' 文件夹。")
            continue
            
        dirty_images = list(dirty_folder.glob("*.jpg")) + list(dirty_folder.glob("*.png")) + list(dirty_folder.glob("*.jpeg"))
        
        if not dirty_images:
            print(f"  [警告] 在 'dirty' 文件夹中没有找到任何“脏图”。")
            continue
            
        dirty_images.sort() # 确保一个一致的 01, 02... 顺序
        
        # 6. 移动并重命名“脏图”
        for j, dirty_file_path in enumerate(dirty_images):
            dirty_ext = dirty_file_path.suffix
            # (例如: S001_dirty_01.jpg, S001_dirty_02.jpg ...)
            new_dirty_name = f"{base_id}{DIRTY_SUFFIX}_{j+1:02d}{dirty_ext}"
            dest_dirty_path = DEST_DIRTY / new_dirty_name
            shutil.copy(str(dirty_file_path), str(dest_dirty_path)) # 使用 copy
            total_dirty_copied += 1
            
        print(f"  (问题) -> {len(dirty_images)} 张“脏图”已重命名并保存 (例如 {new_dirty_name})")

    print("\n--- 成功! ---")
    print(f"所有“成对”数据（`dirty_dataset_collection_plan.md`）已处理完毕。")
    print(f"共复制了 {total_clean_copied} 张“净图”到: {DEST_CLEAN}")
    print(f"共复制了 {total_dirty_copied} 张“脏图”到: {DEST_DIRTY}")
    print(f"\n您现在可以运行 'run_correction_pipeline.py'（`run_correction_pipeline.py`）了。")

if __name__ == "__main__":
    process_folders()