import numpy as np
import cvxpy as cp
from scipy.fftpack import dct

def dct2d_matrix_inverse(patch_size):
    D1 = dct(np.eye(patch_size), norm='ortho')
    return np.kron(D1.T, D1.T)

def reconstruct_from_mask_cvx(image, mask, patch_size, stride, lam=0.01):
    H, W = image.shape
    N = patch_size * patch_size
    Psi_inv = dct2d_matrix_inverse(patch_size)  # Psi^{-1} ∈ R^{N×N}

    recon_img = np.zeros((H, W), dtype=np.float32)
    weight_img = np.zeros((H, W), dtype=np.float32)

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            # reconstruction for single patch
            patch = image[i:i+patch_size, j:j+patch_size]
            patch_mask = mask[i:i+patch_size, j:j+patch_size]

            x = patch.flatten().reshape(-1, 1)
            msk = patch_mask.flatten().astype(bool)

            Phi = np.eye(N)[msk]     
            y = Phi @ x            
            A = Phi @ Psi_inv         

            # Solve sparse code z using LASSO
            z = cp.Variable((N, 1))
            objective = cp.Minimize(0.5 * cp.sum_squares(A @ z - y) + lam * cp.norm1(z))
            prob = cp.Problem(objective)
            prob.solve(solver=cp.SCS, verbose=False)

            if z.value is None:
                continue

            x_hat = Psi_inv @ z.value  # reconstruct patch
            patch_recon = x_hat.reshape((patch_size, patch_size))

            # 累加并加权平均
            recon_img[i:i+patch_size, j:j+patch_size] += patch_recon
            weight_img[i:i+patch_size, j:j+patch_size] += 1

    # normalization
    weight_img[weight_img == 0] = 1
    recon_img /= weight_img
    recon_img = np.clip(recon_img, 0, 1)

    return recon_img
