import sys
import os
import json
import numpy as np
from scipy.io import loadmat, savemat
from algorithms.cvx.cvx import reconstruct_from_mask_cvx
from algorithms.utils.common import evaluate_reconstruction, r_factor_masked

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
ref_image = image

if "snr" in cfg and cfg["snr"] is not None:
    snr_db = cfg["snr"]
    sigma = np.std(image) * 10 ** (-snr_db / 20)
    noise = sigma * np.random.randn(*image.shape)
    image = np.clip(image + noise, 0, 1)

mask = loadmat("sampling_mask.mat")["mask"]
patch_size = cfg["patch_size"]
stride = cfg["stride"]
lam = cfg["lam"]

recon = reconstruct_from_mask_cvx(
    image, mask, patch_size=patch_size, stride=stride, lam=lam
)

savemat(cfg["output_path"], {"recon_img": recon})

psnr_val, ssim_val = evaluate_reconstruction(ref_image, recon)
rfactor_val = r_factor_masked(recon, ref_image, mask)
with open(cfg["metrics_path"], "w") as f:
    json.dump({"psnr": psnr_val, "ssim": ssim_val, "r_factor": rfactor_val}, f)
