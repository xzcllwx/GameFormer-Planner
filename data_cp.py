import os
import random
import shutil
from tqdm import tqdm
def copy_and_record_files(src_dir, dst_dir, record_file):
    # 获取源文件夹中的所有文件
    files = os.listdir(src_dir)
    
    # 随机选择一半文件
    random.shuffle(files)
    half_files = files[:len(files) // 2]
    print(f"Total files: {len(files)}, half files: {len(half_files)}")
    # 确保目标文件夹存在
    os.makedirs(dst_dir, exist_ok=True)
    
    # 复制文件并记录文件名
    with open(record_file, 'w') as f:
        for file in tqdm(half_files, desc="Copying files"):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy(src_file, dst_file)
            f.write(f"{file}\n")
            print(f"Copied {src_file} to {dst_file}")


def copy_remaining_files(src_dir, dst_dir, record_file, remain_file):
    # 获取源文件夹中的所有文件
    files = os.listdir(src_dir)
    
    # 读取记录文件中的文件名
    with open(record_file, 'r') as f:
        recorded_files = set(f.read().splitlines())

    # 过滤掉记录文件中的文件
    remaining_files = [file for file in files if file not in recorded_files]
    print(f"Total files: {len(files)}, remaining files: {len(remaining_files)}")
    # 确保目标文件夹存在
    os.makedirs(dst_dir, exist_ok=True)
    
    # 复制剩余的文件
    with open(remain_file, 'w') as f:
        for file in tqdm(remaining_files, desc="Copying remaining files"):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy(src_file, dst_file)
            f.write(f"{file}\n")
            print(f"Copied {src_file} to {dst_file}")
    # for file in tqdm(remaining_files, desc="Copying remaining files"):
    #     src_file = os.path.join(src_dir, file)
    #     dst_file = os.path.join(dst_dir, file)
    #     shutil.copy(src_file, dst_file)
    #     print(f"Copied {src_file} to {dst_file}")
def compare_text_files(file1_path, file2_path):
    # 读取第一个文件的所有行
    with open(file1_path, 'r') as f1:
        lines1 = set(line.strip() for line in f1)
    
    # 读取第二个文件的所有行
    with open(file2_path, 'r') as f2:
        lines2 = set(line.strip() for line in f2)
    
    # 找出共同的行
    common_lines = lines1.intersection(lines2)
    
    # 输出结果
    if common_lines:
        print(f"发现 {len(common_lines)} 行相同内容")
        # 如果需要查看具体内容，可以取消下面的注释
        # for line in common_lines:
        #     print(line)
    else:
        print("两个文件没有共同的行")
    
    # 输出两个文件的总行数
    print(f"第一个文件总行数: {len(lines1)}")
    print(f"第二个文件总行数: {len(lines2)}")
    print(f"两个文件总行数: {len(lines1) + len(lines2)}")
    
    # 返回共同行的数量和内容
    return len(common_lines), common_lines

def copy_random_files(src_dir, dst_dir, record_file, num_files=1000000):
    # 递归获取源文件夹中的所有文件
    files = []
    for root, _, filenames in os.walk(src_dir):
        for filename in filenames:
            files.append(os.path.relpath(os.path.join(root, filename), src_dir))
    
    # 如果文件总数少于要求的数量，则全部复制
    if len(files) <= num_files:
        selected_files = files
        print(f"Total files: {len(files)}, all will be copied")
    else:
        # 随机选择指定数量的文件
        random.shuffle(files)
        selected_files = files[:num_files]
        print(f"Total files: {len(files)}, randomly selecting {num_files} files")
    
    # 确保目标文件夹存在
    os.makedirs(dst_dir, exist_ok=True)
    
    # 复制文件并记录文件名
    with open(record_file, 'w') as f:
        for file in tqdm(selected_files, desc=f"Copying {num_files} random files"):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            
            # 确保目标子目录存在
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            
            # 复制文件
            shutil.copy2(src_file, dst_file)
            f.write(f"{file}\n")
    
    print(f"Successfully copied {len(selected_files)} files to {dst_dir}")

def delete_random_files(dir_path, num_files=3000000, dry_run=True, record_file=None):
    """
    从指定目录中随机删除指定数量的文件
    
    参数:
    dir_path (str): 目标文件夹路径
    num_files (int): 要删除的文件数量
    dry_run (bool): 如果为True，仅列出要删除的文件但不实际删除
    record_file (str): 记录被删除文件的日志文件路径
    """
    # 递归获取目录中的所有文件
    all_files = []
    print("正在遍历目录...")
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            all_files.append(os.path.join(root, filename))
    
    total_files = len(all_files)
    print(f"目录中共有 {total_files} 个文件")
    
    if total_files <= num_files:
        print(f"警告: 请求删除 {num_files} 个文件，但目录中只有 {total_files} 个文件")
        files_to_delete = all_files
    else:
        # 随机选择要删除的文件
        random.shuffle(all_files)
        files_to_delete = all_files[:num_files]
    
    print(f"将{'测试' if dry_run else ''}删除 {len(files_to_delete)} 个文件")
    
    # 确认删除
    if not dry_run:
        confirm = input(f"确认要删除 {len(files_to_delete)} 个文件? (y/n): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return
    
    # 打开记录文件(如果提供)
    if record_file:
        f = open(record_file, 'w')
    
    # 删除文件
    deleted_count = 0
    failed_count = 0
    
    for file_path in tqdm(files_to_delete, desc=f"{'列出' if dry_run else '删除'}文件"):
        if record_file:
            f.write(f"{file_path}\n")
        
        if not dry_run:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"删除 {file_path} 失败: {e}")
                failed_count += 1
    
    if record_file:
        f.close()
    
    # 打印结果
    if dry_run:
        print(f"测试完成: 将删除 {len(files_to_delete)} 个文件")
    else:
        print(f"删除完成: 成功删除 {deleted_count} 个文件，失败 {failed_count} 个文件")


if __name__ == "__main__":
    src_dir = "/root/data/alstar/womd/scenario/validation_process"  # 替换为源文件夹路径
    dst_dir = "/root/xzcllwx_ws/womd_process/val_40K"  # 替换为目标文件夹路径
    record_file = "/root/xzcllwx_ws/womd_process/womd_1M.txt"  # 替换为记录文件路径
    womd_record_file = "/root/xzcllwx_ws/womd_process/womd_val_40K.txt"  # 替换为记录文件路径
    remain_file = "/root/xzcllwx_ws/womd_process/womd_3M.txt"  # 替换为记录文件路径

    # copy_and_record_files(src_dir, dst_dir, record_file)

    # copy_remaining_files(src_dir, dst_dir, record_file, remain_file)

    # count, common = compare_text_files(record_file, remain_file)
    
    # copy_random_files(src_dir, dst_dir, womd_record_file, num_files=400000)
    
    target_dir = "/root/xzcllwx_ws/womd_process/womd_1M"  # 替换为要删除文件的目录
    log_file = "/root/xzcllwx_ws/womd_process/delete_womd_1M.txt"  # 替换为日志文件路径
    delete_random_files(target_dir, num_files=3228499, dry_run=False, record_file=log_file)

    print("Done!")
