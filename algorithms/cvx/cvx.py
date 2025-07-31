import numpy as np
import cvxpy as cp
from scipy.fftpack import dct

def dct2d_matrix(block_size):
    D1 = dct(np.eye(block_size), norm='ortho')
    return np.kron(D1, D1)

#  recon for single patch (2D DCT + LASSO + mean adjustment)
def reconstruct_patch(patch, sampling_rate=0.3, lam=0.01):
    block_size = patch.shape[0]
    x = patch.flatten().reshape(-1, 1)
    n = x.shape[0]

    D = dct2d_matrix(block_size)
    D_inv = D.T

    m = int(n * sampling_rate)
    sample_idx = np.random.choice(n, m, replace=False)
    A = np.zeros((m, n))
    A[np.arange(m), sample_idx] = 1
    y = A @ x

    A_dct = A @ D_inv
    z = cp.Variable((n, 1))
    objective = cp.Minimize(0.5 * cp.sum_squares(A_dct @ z - y) + lam * cp.norm1(z))
    prob = cp.Problem(objective)
    prob.solve(solver=cp.SCS, verbose=False)

    z_hat = z.value
    x_hat = D_inv @ z_hat
    patch_recon = x_hat.reshape(patch.shape)

    # mean adjustment
    recon_mean = np.mean(patch_recon)
    orig_mean = np.mean(patch)
    mean_shift = recon_mean - orig_mean

    if abs(mean_shift) > 0.02:
        patch_recon = patch_recon - mean_shift

    return np.clip(patch_recon, 0, 1)

def blockwise_reconstruct(image, block_size, sampling_rate):
    H, W = image.shape
    X_recon = np.zeros_like(image)
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            patch = image[i:i+block_size, j:j+block_size]
            if patch.shape != (block_size, block_size):
                continue
            recon = reconstruct_patch(patch, sampling_rate=sampling_rate)
            X_recon[i:i+block_size, j:j+block_size] = recon
    return X_recon
