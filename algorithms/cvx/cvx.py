import numpy as np
import cvxpy as cp
from scipy.fftpack import dct
import time

def dct2d_matrix_inverse(patch_size):
    D1 = dct(np.eye(patch_size), norm='ortho')
    return np.kron(D1.T, D1.T)

#  recon for single patch (2D DCT + LASSO + mean adjustment)
def reconstruct_from_mask_cvx(image, mask, patch_size, stride, lam=0.01):
    H, W = image.shape
    N = patch_size * patch_size
    Psi_inv = dct2d_matrix_inverse(patch_size)  # Psi^{-1} ∈ R^{N×N}

    recon_img = np.zeros((H, W), dtype=np.float32)
    weight_img = np.zeros((H, W), dtype=np.float32)
    t1 = time.time()

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patch_mask = mask[i:i+patch_size, j:j+patch_size]

            x = patch.flatten().reshape(-1, 1)
            msk = patch_mask.flatten().astype(bool)

            Phi = np.eye(N)[msk]        # Φ ∈ R^{m×N}
            y = Phi @ x                 # y ∈ R^{m×1}
            A = Phi @ Psi_inv           # A = Φ · Ψ^{-1}

            # Solve sparse code z using LASSO
            z = cp.Variable((N, 1))
            objective = cp.Minimize(0.5 * cp.sum_squares(A @ z - y) + lam * cp.norm1(z))
            prob = cp.Problem(objective)
            prob.solve(solver=cp.SCS, verbose=False)

            if z.value is None:
                continue

            x_hat = Psi_inv @ z.value  # 重建 patch
            patch_recon = x_hat.reshape((patch_size, patch_size))

            # 累加并加权平均
            recon_img[i:i+patch_size, j:j+patch_size] += patch_recon
            weight_img[i:i+patch_size, j:j+patch_size] += 1

    t2 = time.time()
    print(f"解LASSO耗时: {(t2 - t1)*1000:.2f} ms")

    # 归一化加权图像
    t3 = time.time()
    weight_img[weight_img == 0] = 1
    recon_img /= weight_img
    recon_img = np.clip(recon_img, 0, 1)
    t4 = time.time()
    print(f"recon: {(t4 - t3)*1000:.2f} ms")

    return recon_img
