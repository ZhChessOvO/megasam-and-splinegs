import os
import sys
import torch
sys.path.append(os.path.join(sys.path[0], ".."))
import cv2
import lpips
import numpy as np
from argparse import ArgumentParser
from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render_infer
from PIL import Image
from scene import GaussianModel, Scene, dataset_readers
from utils.graphics_utils import pts2pixel
from utils.main_utils import get_pixels
from utils.image_utils import psnr
from gsplat.rendering import fully_fused_projection
from scene import GaussianModel, Scene, dataset_readers, deformation
import random
import json  # 新增：用于保存相机参数


def normalize_image(img):
    return (2.0 * img - 1.0)[None, ...]


def save_camera_params(cam, save_path):
    """保存相机内参和外参到JSON文件"""
    cam_params = {
        # 外参
        "R": cam.R.tolist(),  # 旋转矩阵
        "T": cam.T.tolist(),  # 平移向量
        # 内参
        "focal_length": cam.focal_length.tolist() if hasattr(cam, 'focal_length') else None,
        "principal_point": cam.principal_point.tolist() if hasattr(cam, 'principal_point') else None,
        "image_size": {
            "width": cam.width if hasattr(cam, 'width') else cam.image_width,
            "height": cam.height if hasattr(cam, 'height') else cam.image_height
        },
        "time": cam.time if hasattr(cam, 'time') else None  # 时间信息（如果有）
    }
    with open(save_path, 'w') as f:
        json.dump(cam_params, f, indent=2)


def training_report(scene: Scene, train_cams, test_cams, renderFunc, background, stage, dataset_type, path):
    test_psnr = 0.0
    torch.cuda.empty_cache()

    validation_configs = ({"name": "test", "cameras": test_cams}, {"name": "train", "cameras": train_cams})
    lpips_loss = lpips.LPIPS(net="alex").cuda()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            lpips_test = 0.0
            run_time = 0.0
            elapsed_time_ms_list = []
            
            # 新增：创建保存图像和相机参数的目录
            img_dir = os.path.join(path, config["name"], "images")
            cam_dir = os.path.join(path, config["name"], "camera_params")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(cam_dir, exist_ok=True)
            
            for idx, viewpoint in enumerate(config["cameras"]):     
                if idx == 0: # warmup iter
                    for _ in range(5):
                        render_pkg = renderFunc(
                            viewpoint, scene.stat_gaussians, scene.dyn_gaussians, background
                        )
                  
                torch.cuda.synchronize()        
                start_event.record()
                render_pkg = renderFunc(
                    viewpoint, scene.stat_gaussians, scene.dyn_gaussians, background
                )
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                elapsed_time_ms_list.append(elapsed_time_ms)
                run_time += elapsed_time_ms

                image = render_pkg["render"]
                image = torch.clamp(image, 0.0, 1.0)
                
                # 保存渲染图像
                img = Image.fromarray(
                    (np.clip(image.permute(1, 2, 0).detach().cpu().numpy(), 0, 1) * 255).astype("uint8")
                )
                img.save(os.path.join(img_dir, f"img_{idx}.png"))
                
                # 新增：保存对应的相机参数
                save_camera_params(
                    viewpoint, 
                    os.path.join(cam_dir, f"cam_{idx}.json")
                )

                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                psnr_test += psnr(image, gt_image, mask=None).mean().double()
                lpips_test += lpips_loss.forward(normalize_image(image), normalize_image(gt_image)).item()

            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            lpips_test /= len(config["cameras"])
            run_time /= len(config["cameras"])
            
            print(
                "\n[ITER {}] Evaluating {}: PSNR {}, LPIPS {}, FPS {}".format(
                    -1, config["name"], psnr_test, lpips_test, 1 / (run_time / 1000)
                )
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint file",
    )
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    # 新增：添加保存路径参数（可选）
    parser.add_argument("--save_dir", type=str, default="output", help="Base directory for saving results")
    # 新增：添加相机位姿NPZ文件路径参数
    parser.add_argument("--camera_npz", type=str, required=True, help="Path to the camera poses NPZ file")

    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    dataset = lp.extract(args)
    hyper = hp.extract(args)
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)
    opt = op.extract(args)

    scene = Scene(
        dataset, dyn_gaussians, stat_gaussians, load_coarse=None
    )  # for other datasets rather than iPhone dataset

    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())  # pose network with instance scaling

    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pp.extract(args)

    # 加载原始相机
    original_test_cams = scene.getTestCameras()
    original_train_cams = scene.getTrainCameras()

    # 1. 加载npz文件中的相机位姿（cam_c2w: 相机到世界的变换矩阵）
    npz_path = args.camera_npz  # 从命令行参数获取NPZ路径
    with np.load(npz_path) as data:
        cam_c2w_all = data["cam_c2w"]  # 形状：(24, 4, 4)
    
    # 2. 分割训练集（前12张）和测试集（后12张）位姿
    cam_c2w_test = cam_c2w_all[32:][:12]   # 测试集位姿（相机到世界）

    # 4. 替换测试相机的位姿（scene.getTestCameras()）
    test_cams = original_test_cams
    # assert len(test_cams) == 12, "测试集相机数量应为12"
    for i, cam in enumerate(test_cams):
        c2w = cam_c2w_test[i]
        w2c = np.linalg.inv(c2w)
        
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        cam.R = R
        cam.T = T

    # 后续处理保持不变
    my_test_cams = [i for i in test_cams]
    viewpoint_stack = [i for i in original_train_cams]

    dyn_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud.ply"))
    stat_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud_static.ply"))
    
    dyn_gaussians.flatten_control_point()
    stat_gaussians.save_ply_compact(os.path.join(args.checkpoint, "compact_point_cloud_static.ply"))
    dyn_gaussians.save_ply_compact_dy(os.path.join(args.checkpoint, "compact_point_cloud.ply"))
        

    dyn_gaussians.load_model(args.checkpoint)
    dyn_gaussians._posenet.eval()
    

    pixels = get_pixels(
        scene.train_camera.dataset[0].metadata.image_size_x,
        scene.train_camera.dataset[0].metadata.image_size_y,
        use_center=True,
    )
    if pixels.shape[-1] != 2:
        raise ValueError("The last dimension of pixels must be 2.")
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    y = (
        pixels[..., 1] - scene.train_camera.dataset[0].metadata.principal_point_y
    ) / dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    x = (
        pixels[..., 0] - scene.train_camera.dataset[0].metadata.principal_point_x
    ) / dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)

    with torch.no_grad():
        for cam in viewpoint_stack:
            time_in = torch.tensor(cam.time).float().cuda()
            pred_R, pred_T = dyn_gaussians._posenet(time_in.view(1, 1))
            R_ = torch.transpose(pred_R, 2, 1).detach().cpu().numpy()
            t_ = pred_T.detach().cpu().numpy()
            cam.update_cam(
                R_[0],
                t_[0],
                local_viewdirs,
                batch_shape,
                dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
            )

    # for view_id in range(len(my_test_cams)):
    #     my_test_cams[view_id].update_cam(
    #         viewpoint_stack[0].R,
    #         viewpoint_stack[0].T,
    #         local_viewdirs,
    #         batch_shape,
    #         dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
    #     )

    # 修改：使用自定义保存路径
    save_path = os.path.join(args.save_dir, args.expname)
    training_report(
        scene,
        viewpoint_stack,
        my_test_cams,
        render_infer,
        background,
        "fine",
        scene.dataset_type,
        save_path
    )