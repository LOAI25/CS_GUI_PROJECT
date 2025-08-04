import sys
import os
import json
import numpy as np
from scipy.io import loadmat, savemat
from algorithms.cvx.cvx import reconstruct_from_mask

from algorithms.common import evaluate_reconstruction

# path
print("start working")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open("config.json", "r") as f:
    cfg = json.load(f)

mat_path = "temp_input.mat"
if not os.path.exists(mat_path):
    raise FileNotFoundError("Can't find temp_input.mat")

mat_data = loadmat(mat_path)
image = mat_data["input_image"]
print("temp is loaded")
mask = loadmat("sampling_mask.mat")["mask"]
print("mask is loaded")

# recon
recon = reconstruct_from_mask(
    image,
    mask,
    block_size=cfg["block_size"],
    stride=cfg["blk_len"],  # 可以从 config.json 加个 stride 配置
    lam=0.01,
    min_samples=5
)

savemat(cfg["output_path"], {"recon_img": recon})

# metrics
psnr_val, ssim_val = evaluate_reconstruction(image, recon)

with open(cfg["metrics_path"], "w") as f:
    json.dump({"psnr": psnr_val, "ssim": ssim_val}, f)

# delete temporary file
if os.path.exists(mat_path):
    os.remove(mat_path)
