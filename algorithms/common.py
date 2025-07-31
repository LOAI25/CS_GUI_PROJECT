import numpy as np
import hyperspy.api as hs
from scipy.io import savemat
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

def load_hdf5_image(path, slice_index=0):
    """
    Load an HDF5 image using Hyperspy format and return a normalized image.
    """
    signal = hs.load(path, reader='HSPY')
    data = signal.data.astype(np.float64)

    if data.ndim == 3:
        image = data[:, :, slice_index]
    else:
        image = data

    return (image - image.min()) / (image.max() - image.min())


def get_hdf5_image_dims(path):
    """
    Return the dimensions of an HDF5 image (without normalization).
    Can be used to determine whether the image is 2D or 3D.
    """
    signal = hs.load(path, reader='HSPY')
    data = signal.data
    return data.shape 


def save_temp_mat(image, mat_path="temp_input.mat"):
    """
    Save the image to a temporary .mat file using the key 'input_image'.
    To ensure compatibility between MATLAB and Python, the key is standardized to 'input_image'.
    """
    savemat(mat_path, {"input_image": image})


def delete_temp_mat(mat_path="temp_input.mat"):
    """
    Delete the temporary .mat file.
    """
    if os.path.exists(mat_path):
        os.remove(mat_path)
        print(f"🧹 已删除临时文件 {mat_path}")


def evaluate_reconstruction(original, reconstructed):
    """
    Compute PSNR and SSIM (both images must be normalized to the range [0, 1]).
    """
    psnr_val = psnr(original, reconstructed, data_range=1.0)
    ssim_val = ssim(original, reconstructed, data_range=1.0)
    return psnr_val, ssim_val
