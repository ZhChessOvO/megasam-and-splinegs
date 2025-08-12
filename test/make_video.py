import cv2
import os
import numpy as np
from glob import glob

def add_white_border(image, target_width, target_height):
    height, width = image.shape[:2]
    top = (target_height - height) // 2
    bottom = target_height - height - top
    left = (target_width - width) // 2
    right = target_width - width - left
    return cv2.copyMakeBorder(
        image, 
        top, bottom, left, right, 
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]
    )

def images_to_video():
    base_path = "/share/czh/splinegs_0811"
    scenes = ["Balloon1", "Balloon2", "Playground", "Skating", "Truck"]
    output_video = os.path.join(base_path, "train_video.mp4")
    fps = 3  
    image_paths = []
    for scene in scenes:
        scene_image_path = os.path.join(
            base_path, 
            scene, 
            "fine_render", 
            "train", 
            "images", 
            "0*.jpg"
        )
        scene_images = sorted(glob(scene_image_path))
        # 过滤带 decomp 的图像
        for img_path in scene_images:
            if "decomp" not in os.path.basename(img_path):
                image_paths.append(img_path)
        # 检查过滤后每个场景图像数量（可选，用于验证过滤效果）
        filtered_count = len([p for p in scene_images if "decomp" not in os.path.basename(p)])
        if filtered_count != 12:
            print(f"警告: 场景 {scene} 过滤后有 {filtered_count} 张图像，预期12张")

    print(f"总共收集到 {len(image_paths)} 张图像")
    if not image_paths:
        print("错误: 未找到任何有效图像")
        return

    max_width = 0
    max_height = 0
    valid_images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}，已跳过")
            continue
        height, width = img.shape[:2]
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        valid_images.append(img_path)

    print(f"检测到最大图像尺寸: {max_width}x{max_height}，将以此为标准尺寸")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video, fourcc, fps, (max_width, max_height))

    for i, img_path in enumerate(valid_images):
        img = cv2.imread(img_path)
        img_with_border = add_white_border(img, max_width, max_height)
        out.write(img_with_border)
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(valid_images)} 张图像")

    out.release()
    cv2.destroyAllWindows()

    if os.path.exists(output_video):
        print(f"视频已成功生成: {output_video}")
        print(f"视频参数: 帧率 {fps}fps, 分辨率 {max_width}x{max_height}")
    else:
        print("错误: 视频生成失败")

if __name__ == "__main__":
    images_to_video()
