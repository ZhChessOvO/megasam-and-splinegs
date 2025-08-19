import os
import numpy as np
from PIL import Image

def convert_rgb_to_bw(mask_dir, threshold=50):
    """
    将目录中所有RGB格式的掩码（蓝色有效区域，黑色背景）转换为黑白掩码
    并替换原文件
    """
    # 获取目录中所有PNG文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    if not mask_files:
        print(f"警告：在 {mask_dir} 中未找到PNG文件")
        return
    
    for filename in mask_files:
        # 构建完整路径
        file_path = os.path.join(mask_dir, filename)
        
        try:
            # 打开图像并转换为numpy数组
            with Image.open(file_path) as img:
                img_np = np.array(img)
            
            # 提取蓝色通道（RGB中的第三个通道）
            blue_channel = img_np[:, :, 2]
            
            # 应用阈值：蓝色区域变为白色(255)，黑色区域保持黑色(0)
            bw_mask = (blue_channel > threshold).astype(np.uint8) * 255
            
            # 转换为单通道图像并保存（替换原文件）
            bw_img = Image.fromarray(bw_mask)
            bw_img.save(file_path)
            
            print(f"已转换并替换: {filename}")
        
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 掩码目录路径
    mask_directory = "/share/czh/stereo_0815/motion_masks"
    
    # 转换阈值（可根据实际蓝色深浅调整）
    threshold = 50  # 蓝色越深，阈值可设越低
    
    print(f"开始转换 {mask_directory} 中的掩码文件...")
    convert_rgb_to_bw(mask_directory, threshold)
    print("所有掩码转换完成")
