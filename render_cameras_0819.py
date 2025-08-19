import os
import sys
import torch
import cv2
import numpy as np
from argparse import ArgumentParser
from PIL import Image
sys.path.append(os.path.join(sys.path[0], ".."))  # 假设与原代码库结构一致

# 从原项目导入必要组件
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from gaussian_renderer import render_infer
from scene import GaussianModel, Scene
from utils.graphics_utils import pts2pixel
from utils.main_utils import get_pixels


def render_cameras(scene, cameras, render_func, background, output_path):
    """
    渲染指定相机列表并保存为PNG图片
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 预热渲染（确保CUDA初始化完成）
    if cameras:
        for _ in range(5):
            render_func(cameras[0], scene.stat_gaussians, scene.dyn_gaussians, background)
        
        # 正式渲染每个相机
        for idx, viewpoint in enumerate(cameras):
            print(f"Rendering camera {idx}/{len(cameras)}...")
            
            # 执行渲染
            with torch.no_grad():
                render_pkg = render_func(viewpoint, scene.stat_gaussians, scene.dyn_gaussians, background)
            
            # 处理渲染结果
            image = render_pkg["render"]
            image = torch.clamp(image, 0.0, 1.0)
            
            # 转换为可保存的图像格式
            img_np = (
                np.clip(image.permute(1, 2, 0).detach().cpu().numpy(), 0, 1) * 255
            ).astype("uint8")
            img = Image.fromarray(img_np)
            
            # 保存图像
            save_path = os.path.join(output_path, f"render_{idx:04d}.png")
            img.save(save_path)
            print(f"Saved to {save_path}")


def main():
    # 解析命令行参数
    parser = ArgumentParser(description="Render images from specified cameras")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Directory to save rendered images"
    )
    parser.add_argument(
        "--camera-type", 
        type=str, 
        choices=["train", "test", "both"], 
        default="test",
        help="Which cameras to render (train/test/both)"
    )
    parser.add_argument("--expname", type=str, default="", help="Experiment name for output subdirectory")
    parser.add_argument("--configs", type=str, default="", help="Path to config file")
    
    args = parser.parse_args()
    
    # 处理配置文件
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    # 构建输出路径
    if args.expname:
        output_path = os.path.join(args.output, args.expname)
    else:
        output_path = args.output
    
    # 初始化场景和高斯模型
    dataset = lp.extract(args)
    hyper = hp.extract(args)
    
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)
    
    scene = Scene(
        dataset, dyn_gaussians, stat_gaussians, load_coarse=None
    )
    
    # 加载模型参数
    dyn_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud.ply"))
    stat_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud_static.ply"))
    dyn_gaussians.load_model(args.checkpoint)
    dyn_gaussians._posenet.eval()
    
    # 准备背景
    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 准备相机内参和方向
    train_cams = scene.getTrainCameras()
    if train_cams:
        # 获取像素和视角方向
        pixels = get_pixels(
            train_cams[0].dataset[0].metadata.image_size_x,
            train_cams[0].dataset[0].metadata.image_size_y,
            use_center=True,
        )
        batch_shape = pixels.shape[:-1]
        pixels = np.reshape(pixels, (-1, 2))
        
        # 计算视角方向
        focal = dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
        y = (pixels[..., 1] - train_cams[0].dataset[0].metadata.principal_point_y) / focal
        x = (pixels[..., 0] - train_cams[0].dataset[0].metadata.principal_point_x) / focal
        viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)
        
        # 更新相机参数
        with torch.no_grad():
            for cam in train_cams:
                time_in = torch.tensor(cam.time).float().cuda()
                pred_R, pred_T = dyn_gaussians._posenet(time_in.view(1, 1))
                R_ = torch.transpose(pred_R, 2, 1).detach().cpu().numpy()
                t_ = pred_T.detach().cpu().numpy()
                cam.update_cam(
                    R_[0], t_[0], local_viewdirs, batch_shape, focal
                )
        
        # 处理测试相机（使用第一个训练相机的位姿）
        test_cams = scene.getTestCameras()
        for cam in test_cams:
            cam.update_cam(
                train_cams[0].R, train_cams[0].T,
                local_viewdirs, batch_shape, focal
            )
    else:
        raise ValueError("No training cameras found in dataset")
    
    # 选择要渲染的相机
    cameras_to_render = []
    if args.camera_type in ["train", "both"]:
        cameras_to_render.extend(train_cams)
    if args.camera_type in ["test", "both"]:
        cameras_to_render.extend(test_cams)
    
    if not cameras_to_render:
        raise ValueError("No cameras selected for rendering")
    
    # 执行渲染
    render_cameras(
        scene=scene,
        cameras=cameras_to_render,
        render_func=render_infer,
        background=background,
        output_path=output_path
    )


if __name__ == "__main__":
    main()