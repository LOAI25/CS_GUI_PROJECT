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
        image = data[slice_index, :, :]
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


def evaluate_reconstruction(original, reconstructed):
    """
    Compute PSNR and SSIM (both images must be normalized to the range [0, 1]).
    """
    psnr_val = psnr(original, reconstructed, data_range=1.0)
    ssim_val = ssim(original, reconstructed, data_range=1.0)
    return psnr_val, ssim_val


def generate_sampling_mask(H, W, sampling_rate, method="random", seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = int(round(H * W * sampling_rate))
    mask = np.zeros((H, W), dtype=bool)

    if method.lower() == 'random':
        idx = np.random.choice(H * W, N, replace=False)
        mask.flat[idx] = True
        return mask

    elif method.lower() == 'linehop':
        # === Estimate lane count based on sampling rate ===
        N_target = N
        lane_count = int(round(H * sampling_rate))

        best_err = float('inf')
        best_sequence = []

        for max_h in range(2, int(H / lane_count) + 6):
            for _ in range(1000):
                candidate = []
                total = 0
                while total < H:
                    h = np.random.randint(2, max_h + 1)
                    if total + h > H:
                        break
                    candidate.append(h)
                    total += h

                err = abs(len(candidate) - lane_count)
                if abs(total - H) <= 1 and err < best_err:
                    best_err = err
                    best_sequence = candidate

        lane_heights = np.random.permutation(best_sequence)

        mask = np.zeros((H, W), dtype=bool)
        y_start = 0
        x_dir = 1
        count = 0

        for h in lane_heights:
            if y_start + h > H:
                break

            y_range = np.arange(y_start, y_start + h)
            x_path = np.arange(W) if x_dir == 1 else np.arange(W-1, -1, -1)

            y_path = np.zeros(W, dtype=int)
            y_path[0] = np.random.choice(y_range)

            for k in range(1, W):
                prev_y = y_path[k - 1]
                candidates = [prev_y + d for d in [-1, 0, 1] if (prev_y + d) in y_range]
                y_path[k] = np.random.choice(candidates)

            for k in range(W):
                mask[y_path[k], x_path[k]] = True
                count += 1

            y_start += h
            x_dir *= -1

        # === Adjust sampling count ===
        current_count = np.count_nonzero(mask)
        if current_count > N_target:
            idx = np.flatnonzero(mask)
            idx = np.random.permutation(idx)[:N_target]
            mask[:] = False
            mask.flat[idx] = True
        elif current_count < N_target:
            rest = np.flatnonzero(~mask)
            idx = np.random.permutation(rest)[:(N_target - current_count)]
            mask.flat[idx] = True

        return mask

    else:
        raise ValueError(f"Unknown sampling method: {method}")


def save_mask_to_mat(mask, save_path="sampling_mask.mat"):
    """
    保存 mask 为 .mat 文件，确保为 uint8 类型（0/1），兼容 MATLAB。
    
    参数:
        mask: numpy.ndarray (bool 或 int 类型)，尺寸为 (H, W)
        save_path: 保存路径，默认为 "sampling_mask.mat"
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask 必须是 numpy ndarray")
    if mask.ndim != 2:
        raise ValueError("mask 必须是二维图像掩码")

    # 转换为 uint8 (0 or 1)
    mat_mask = mask.astype(np.uint8)

    # 保存为 .mat 文件，变量名叫 'mask'
    savemat(save_path, {"mask": mat_mask})
    print(f"Mask 保存到: {os.path.abspath(save_path)}")