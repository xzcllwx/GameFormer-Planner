
import os
def get_tokens_from_filenames(directory):
    tokens = set()
    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            # 假设文件名格式为 {map_name}_{token}.npz
            token = filename.split('_')[-1].replace('.npz', '')
            tokens.add(token)
    return tokens

if __name__ == "__main__":
    directory = "/root/data/alstar/nuplan/dataset/nuplan-v1.1/splits/train_1M"  # 替换为你的文件夹路径
    tokens = get_tokens_from_filenames(directory)
    
    # 将 tokens 保存到文件中
    output_file = "/root/xzcllwx_ws/tokens.txt"  # 替换为你想保存的文件路径
    with open(output_file, 'w') as f:
        for token in tokens:
            f.write(f"{token}\n")
    
    print(f"Tokens have been saved to {output_file}, size: {len(tokens)}")