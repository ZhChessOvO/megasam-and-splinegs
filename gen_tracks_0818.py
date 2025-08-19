""" Code borrowed from 
https://github.com/vye16/shape-of-motion/blob/main/preproc/compute_tracks_torch.py
"""
import argparse
import glob
import os

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from cotracker.utils.visualizer import Visualizer

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def read_video(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.concatenate([np.array(Image.open(frame_path)).transpose(2, 0, 1)[None, None] for frame_path in frame_paths], axis=1)
    video = torch.from_numpy(video).float()
    return video

def read_mask(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    # 读取RGB掩码并转换为合适的格式
    video = np.concatenate([np.array(Image.open(frame_path)).transpose(2, 0, 1)[None, None] for frame_path in frame_paths], axis=1)
    video = torch.from_numpy(video).float()
    return video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--mask_dir", type=str, required=True, help="mask dir")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--is_static", action="store_true")
    parser.add_argument("--grid_size", type=int, default=100, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    # 添加掩码阈值参数
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=50,
        help="Threshold for blue channel in mask (default: 50)"
    )
    args = parser.parse_args()

    folder_path = args.image_dir
    mask_dir = args.mask_dir
    frame_names = [
        os.path.basename(f) for f in sorted(glob.glob(os.path.join(folder_path, "*")))
    ]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)

    done = True
    for t in range(len(frame_names)):
        for j in range(len(frame_names)):
            name_t = os.path.splitext(frame_names[t])[0]
            name_j = os.path.splitext(frame_names[j])[0]
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            if not os.path.exists(out_path):
                done = False
                break
    print(f"{done}")
    if done:
        print("Already done")
        return

    ## Load model
    # model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEFAULT_DEVICE)
    # 强制使用本地缓存，跳过在线验证
    model = torch.hub.load(
        "facebookresearch/co-tracker", 
        "cotracker3_offline",
        source='github',
        force_reload=False,  # 不强制重新下载
        skip_validation=True  # 跳过在线验证（关键）
    ).to(DEFAULT_DEVICE)
    
    video = read_video(folder_path).to(DEFAULT_DEVICE)
    
    masks = read_mask(mask_dir).to(DEFAULT_DEVICE)
    
    # 处理RGB掩码：提取蓝色通道并转换为二值掩码
    # 提取蓝色通道 (RGB中的第三个通道)
    masks = masks[:, :, 2, :, :]  # 现在是单通道
    
    # 应用阈值：蓝色区域变为1.0，黑色区域变为0.0
    masks = (masks > args.mask_threshold).float()

    # print("mask:",masks.shape)
    # print("video:",video.shape)
    
    if args.is_static:
        masks = 1.0 - masks  # 如果是静态掩码，反转有效区域
    
    _, num_frames, height, width = masks.shape  # 更新掩码形状信息
    vis = Visualizer(save_dir=os.path.join(out_dir, "vis"), pad_value=120, linewidth=3)

    for t in tqdm(range(num_frames), desc="query frames"):
        name_t = os.path.splitext(frame_names[t])[0]
        file_matches = glob.glob(f"{out_dir}/{name_t}_*.npy")
        if len(file_matches) == num_frames:
            print(f"Already computed tracks with query {t} {name_t}")
            continue

        current_mask = masks[:,t].unsqueeze(1)

        # print("current mask:", current_mask.shape)
        # print("video:", video.shape)

        # 获取当前帧的掩码并调整维度
        # current_mask = masks[:, t].unsqueeze(1).unsqueeze(1)  # 调整为符合模型要求的维度
        
        # # 确保掩码与视频尺寸匹配
        # current_mask = torch.nn.functional.interpolate(
        #     current_mask,
        #     size=(video.shape[-2], video.shape[-1]),  # 匹配视频的高和宽
        #     mode="nearest"
        # )
        
        start_pred = None
        
        for j in range(num_frames):
            if j > t:
                current_video = video[:,t:j+1]
            elif j < t:
                current_video = torch.flip(video[:,j:t+1], dims=(1,)) # reverse
            else:
                continue
                # current_video = video[:,t:t+1]
            
        
            pred_tracks, pred_visibility = model(
                current_video,
                grid_size=args.grid_size,
                grid_query_frame=0,
                backward_tracking=False,
                segm_mask=current_mask
            )
            

            pred = torch.cat([pred_tracks, pred_visibility.unsqueeze(-1)], dim=-1)
            current_pred = pred[0,-1]
            start_pred = pred[0,0]

            # save
            name_j = os.path.splitext(frame_names[j])[0]
            np.save(f"{out_dir}/{name_t}_{name_j}.npy", current_pred.cpu().numpy())
            
            # visualize
            # vis.visualize(current_video, pred_tracks, pred_visibility, filename=f"{name_t}_{name_j}")
            
        np.save(f"{out_dir}/{name_t}_{name_t}.npy", start_pred.cpu().numpy())



if __name__ == "__main__":
    main()
