import os
import re

def rename_uni_depth_files(directory):
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误：目录 {directory} 不存在")
        return
    
    # 正则表达式匹配 train_xxx.npy 或 train_xxx.png 格式
    # 捕获数字部分和文件扩展名
    pattern = r'^train_(\d+)\.(npy|png)$'
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否匹配模式
        match = re.match(pattern, filename)
        if match:
            # 提取数字部分和文件扩展名
            number = match.group(1)
            ext = match.group(2)
            
            # 构建新文件名
            new_filename = f"{number}.{ext}"
            
            # 构建完整路径
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # 避免文件名重复（虽然理论上不会出现）
            if os.path.exists(new_path):
                print(f"警告：{new_filename} 已存在，跳过 {filename}")
                continue
            
            # 执行重命名
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_filename}")

if __name__ == "__main__":
    # 目标目录
    target_dir = "/share/czh/stereo_0815/images_2"
    rename_uni_depth_files(target_dir)
    print("重命名操作完成")
