import os
import numpy as np
from PIL import Image

def convert_single_png_to_bw(folder_path, threshold=50):
    """
    将单个文件夹下的000.png转换为黑白格式
    :param folder_path: 文件夹路径，如包含000.png的"000"、"001"等文件夹路径
    :param threshold: 蓝色通道的阈值，用于判断是否为有效区域
    """
    png_file_path = os.path.join(folder_path, "000.png")
    if not os.path.exists(png_file_path):
        print(f"{folder_path} 下未找到000.png文件，跳过该文件夹")
        return
    
    try:
        # 打开图像并转换为numpy数组
        with Image.open(png_file_path) as img:
            img_np = np.array(img)
        
        # 提取蓝色通道（RGB中的第三个通道，索引为2）
        blue_channel = img_np[:, :, 2]
        
        # 应用阈值：蓝色区域转为白色(255)，其他转为黑色(0)
        bw_mask = (blue_channel > threshold).astype(np.uint8) * 255
        
        # 转换为单通道图像并保存（替换原文件）
        bw_img = Image.fromarray(bw_mask)
        bw_img.save(png_file_path)
        print(f"已转换 {png_file_path}")
    except Exception as e:
        print(f"处理 {png_file_path} 时出错: {str(e)}")

def batch_convert_pngs(root_dir, threshold=50):
    """
    批量遍历root_dir下的子文件夹（如000、001等），转换其中的000.png
    :param root_dir: 包含多个子文件夹（000、001等）的根目录
    :param threshold: 蓝色通道的阈值
    """
    for sub_folder in os.listdir(root_dir):
        sub_folder_path = os.path.join(root_dir, sub_folder)
        if os.path.isdir(sub_folder_path):
            convert_single_png_to_bw(sub_folder_path, threshold)

if __name__ == "__main__":
    # 请替换为实际包含000、001等子文件夹的根目录路径
    root_directory = "/share/czh/stereo_0815/instance_mask"  
    threshold = 50  # 可根据实际情况调整蓝色通道的阈值
    
    print(f"开始批量转换 {root_directory} 下子文件夹中的000.png...")
    batch_convert_pngs(root_directory, threshold)
    print("所有转换操作完成")