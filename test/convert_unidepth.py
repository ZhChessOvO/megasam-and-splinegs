#!/usr/bin/env python3
import os, glob, numpy as np
from PIL import Image

IN_ROOT  = '/share/czh/nvidia_megasam_preprocess/UniDepth'
OUT_ROOT = '/share/czh/nvidia_megasam'

# 全局统计一次，保证 12 张图同比例拉伸
def global_min_max(npz_files):
    all_vals = [np.load(p)['depth'] for p in npz_files]
    whole = np.concatenate([d.ravel() for d in all_vals])
    return whole.min(), whole.max()

for scene in ['Balloon1', 'Balloon2', 'Playground', 'Skating', 'Truck']:
    npz_files = sorted(glob.glob(os.path.join(IN_ROOT, scene, '*.npz')))
    out_dir   = os.path.join(OUT_ROOT, scene, 'uni_depth')
    os.makedirs(out_dir, exist_ok=True)

    dmin, dmax = global_min_max(npz_files)   # 全局 min / max
    print(scene, 'global min:', dmin, 'max:', dmax)

    for idx, npz_path in enumerate(npz_files):
        d = np.load(npz_path)['depth'].astype(np.float32)

        # 1) .npy 保持原始深度
        np.save(os.path.join(out_dir, f'{idx:03d}.npy'), d)

        # 2) PNG 可视：全局归一化 -> 0-255
        d_norm = (d - dmin) / (dmax - dmin)        # 0~1
        d_uint8 = (d_norm * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(d_uint8, mode='L').save(
            os.path.join(out_dir, f'{idx:03d}.png')
        )

print('All scenes converted with visible PNGs.')