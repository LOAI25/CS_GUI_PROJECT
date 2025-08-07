import sys
import os
import json
import numpy as np
from scipy.io import loadmat, savemat
from algorithms.cvx.cvx import reconstruct_from_mask
from algorithms.common import evaluate_reconstruction

# 确保项目根路径在 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 读取配置
with open("config.json", "r") as f:
    cfg = json.load(f)

# 加载输入图像
mat_path = "temp_input.mat"
if not os.path.exists(mat_path):
    raise FileNotFoundError("Can't find temp_input.mat")
mat_data = loadmat(mat_path)
image = mat_data["input_image"]

# 加载采样 mask
mask = loadmat("sampling_mask.mat")["mask"]

# 必填参数（直接取值）
patch_size = cfg["patch_size"]
stride = cfg["stride"]
lam = cfg["lam"]

# 执行重建
recon = reconstruct_from_mask(
    image,
    mask,
    patch_size=patch_size,
    stride=stride,
    lam=lam,
    min_samples=5
)

# 保存结果
savemat(cfg["output_path"], {"recon_img": recon})

# 计算并保存评价指标
psnr_val, ssim_val = evaluate_reconstruction(image, recon)
with open(cfg["metrics_path"], "w") as f:
    json.dump({"psnr": psnr_val, "ssim": ssim_val}, f)

# 删除临时文件
if os.path.exists(mat_path):
    os.remove(mat_path)
