import sys
import os
import json
import numpy as np
from scipy.io import loadmat, savemat
from algorithms.CoSaMP.CoSaMP import reconstruct_from_mask_cosamp
from algorithms.common import evaluate_reconstruction

# 确保项目根路径在 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === 加载配置 ===
with open("config.json", "r") as f:
    cfg = json.load(f)

# === 加载图像 & mask ===
image = loadmat("temp_input.mat")["input_image"]
mask = loadmat("sampling_mask.mat")["mask"]

# === 基本参数 ===
patch_size = cfg["patch_size"]
stride = cfg["stride"]
max_iters = cfg["max_iters"]
K = cfg["K"]

# === 调用 CoSaMP 重建函数 ===
recon = reconstruct_from_mask_cosamp(
    image,
    mask,
    patch_size=patch_size,
    stride=stride,
    max_iters=max_iters,
    K=K
)

# === 保存重建图像 ===
savemat(cfg["output_path"], {"recon_img": recon})

# === 保存评价指标 ===
psnr_val, ssim_val = evaluate_reconstruction(image, recon)
with open(cfg["metrics_path"], "w") as f:
    json.dump({"psnr": psnr_val, "ssim": ssim_val}, f)
