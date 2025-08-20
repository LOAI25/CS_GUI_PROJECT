import numpy as np
from scipy.fftpack import dct
import jax.numpy as jnp
from cr.sparse.pursuit.cosamp import matrix_solve


def dct2d_matrix_inverse(patch_size):
    D = dct(np.eye(patch_size), norm='ortho')
    return np.kron(D.T, D.T)


def reconstruct_from_mask_cosamp(image, mask, patch_size, stride, max_iters=200, K=30):
    H, W = image.shape
    N = patch_size * patch_size
    Psi_inv = dct2d_matrix_inverse(patch_size)

    recon_img = np.zeros((H, W), dtype=np.float32)
    weight_img = np.zeros((H, W), dtype=np.float32)

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patch_mask = mask[i:i+patch_size, j:j+patch_size]

            x = patch.flatten()
            msk = patch_mask.flatten().astype(bool)


            Phi = np.eye(N)[msk]  # Φ ∈ R^{m×N}
            y = Phi @ x
            A = Phi @ Psi_inv

            # CoSaMP sparse recovery
            result = matrix_solve(jnp.array(A), jnp.array(y), max_iters=max_iters, K=K)

            z = jnp.zeros(N).at[result.I].set(result.x_I)
            x_hat = Psi_inv @ np.array(z)
            patch_recon = x_hat.reshape((patch_size, patch_size))

            recon_img[i:i+patch_size, j:j+patch_size] += patch_recon
            weight_img[i:i+patch_size, j:j+patch_size] += 1

    weight_img[weight_img == 0] = 1
    recon_img /= weight_img
    recon_img = np.clip(recon_img, 0, 1)
    return recon_img
