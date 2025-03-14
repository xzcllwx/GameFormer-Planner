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

if __name__ == "__main__":
    src_dir = "/root/data/alstar/nuplan/dataset/nuplan-v1.1/splits/train"  # 替换为源文件夹路径
    dst_dir = "/root/xzcllwx_ws/nuplan_dataset_process/train_half"  # 替换为目标文件夹路径
    record_file = "/root/xzcllwx_ws/nuplan_dataset_process/cp_half.txt"  # 替换为记录文件路径
    remain_file = "/root/xzcllwx_ws/nuplan_dataset_process/cp_remain.txt"  # 替换为记录文件路径

    # copy_and_record_files(src_dir, dst_dir, record_file)

    # copy_remaining_files(src_dir, dst_dir, record_file, remain_file)

    count, common = compare_text_files(record_file, remain_file)
    print("Done!")
