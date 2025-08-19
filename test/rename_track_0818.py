import os
import re

def rename_files(directory):
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"错误：目录 {directory} 不存在")
        return
    
    # 正则表达式匹配文件名格式 train_xxx_train_yyy.npy
    pattern = r'^train_(\d+)_train_(\d+)\.npy$'
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否匹配模式
        match = re.match(pattern, filename)
        if match:
            # 提取两个数字部分
            num1 = match.group(1)
            num2 = match.group(2)
            
            # 构建新文件名
            new_filename = f"{num1}_{num2}.npy"
            
            # 构建完整的旧路径和新路径
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # 重命名文件
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_filename}")

if __name__ == "__main__":
    # 目标目录路径
    target_directory = "/share/czh/stereo_0815/bootscotracker_static"
    rename_files(target_directory)
    print("重命名完成")
    