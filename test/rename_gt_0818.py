import os
import re

def rename_gt_files(directory):
    if not os.path.exists(directory):
        print(f"错误：目录 {directory} 不存在")
        return
    
    # 匹配 val_xxx.png 格式
    pattern = r'^val_(\d+)\.png$'
    
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            num = match.group(1)
            new_filename = f"v000_t{num}.png"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            if os.path.exists(new_path):
                print(f"警告：{new_filename} 已存在，跳过 {filename}")
                continue
            
            os.rename(old_path, new_path)
            print(f"{filename} → {new_filename}")

if __name__ == "__main__":
    target_dir = "/share/czh/stereo_0815/gt"
    rename_gt_files(target_dir)
    print("重命名完成")