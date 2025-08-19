import os
from PIL import Image

# 定义原始数据路径和目标路径
src_base_path = "/share/czh/stereo/"
train_target_path = "/share/czh/stereo_0815/images_2/"
val_target_path = "/share/czh/stereo_0815/gt/"

# 确保目标文件夹存在
os.makedirs(train_target_path, exist_ok=True)
os.makedirs(val_target_path, exist_ok=True)

def process_folder(src_folder, target_folder, prefix):
    """
    处理文件夹中的JPG文件，按文件名排序后保存为指定格式的PNG文件
    """
    # 获取所有JPG文件并排序
    jpg_files = [f for f in os.listdir(src_folder) if f.lower().endswith(".jpg")]
    # 按文件名从小到大排序
    jpg_files.sort()
    
    # 处理每个文件
    for idx, file in enumerate(jpg_files):
        src_file_path = os.path.join(src_folder, file)
        try:
            with Image.open(src_file_path) as img:
                # 构造目标文件名，确保3位数字格式
                target_file_name = f"{prefix}_{str(idx).zfill(3)}.png"
                target_file_path = os.path.join(target_folder, target_file_name)
                img.save(target_file_path)
            print(f"已处理: {file} -> {target_file_name}")
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")

# 处理train文件夹
train_src_path = os.path.join(src_base_path, "train")
process_folder(train_src_path, train_target_path, "train")

# 处理val文件夹
val_src_path = os.path.join(src_base_path, "val")
process_folder(val_src_path, val_target_path, "val")

print("所有文件处理完成")
    