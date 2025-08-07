clear; clc;

% === 切换到项目根目录 ===
cd(fullfile('..', '..'));
addpath(fileparts(mfilename('fullpath')));

% === 加载配置 ===
fid = fopen('config.json');
if fid == -1
    error("config.json can't be opened");
end
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
cfg = jsondecode(str);

% === 参数提取 ===
patch_size = cfg.patch_size;
stride     = cfg.stride;

% === 加载输入图像 ===
if ~isfile('temp_input.mat')
    error("Can't find temp_input.mat");
end
S = load('temp_input.mat');
img2d = S.input_image;

% === 加载采样 mask ===
if ~isfile('sampling_mask.mat')
    error("Can't find sampling_mask.mat");
end
mask_data = load('sampling_mask.mat');
if ~isfield(mask_data, 'mask')
    error("sampling_mask.mat does not contain variable 'mask'");
end
global_mask = logical(mask_data.mask);

% === 可选：全图添加噪声 ===
if isfield(cfg, 'snr') && ~isempty(cfg.snr)
    sigma = std(img2d(:)) * 10^(-cfg.snr / 20);
    img2d = img2d + sigma * randn(size(img2d));
    img2d = min(max(img2d, 0), 1);  % clip 到 [0, 1]
    fprintf('[INFO] Added global noise with SNR = %.1f dB\n', cfg.snr);
else
    fprintf('[INFO] No noise added (clean input)\n');
end

% === 初始化 ===
[H, W] = size(img2d);
N = patch_size^2;
D = dctmtx(patch_size);
Psi2D = kron(D', D');

recon_img = zeros(H, W);
count_map = zeros(H, W);

% === 加载 SPGL1 工具包 ===
opts = spgSetParms('verbosity', 0);

fprintf("=== SPGL1 Reconstruction with stride=%d, patch_size=%d ===\n", stride, patch_size);

% === 主循环 ===
for i = 1:stride:(H - patch_size + 1)
    for j = 1:stride:(W - patch_size + 1)
        patch = img2d(i:i+patch_size-1, j:j+patch_size-1);
        patch_mask = global_mask(i:i+patch_size-1, j:j+patch_size-1);

        x = patch(:);
        msk = patch_mask(:);

        if sum(msk) < 5
            continue;
        end

        Phi = eye(N); Phi = Phi(msk, :);
        A = Phi * Psi2D;
        y = Phi * x;

        % === 调用 SPGL1 ===
        theta_hat = spg_bp(A, y, opts);
        x_hat = Psi2D * theta_hat;
        patch_hat = reshape(x_hat, patch_size, patch_size);
        patch_hat = min(max(patch_hat, 0), 1);  % clip

        recon_img(i:i+patch_size-1, j:j+patch_size-1) = ...
            recon_img(i:i+patch_size-1, j:j+patch_size-1) + patch_hat;
        count_map(i:i+patch_size-1, j:j+patch_size-1) = ...
            count_map(i:i+patch_size-1, j:j+patch_size-1) + 1;
    end
end

% === 重叠区域归一化 ===
count_map(count_map == 0) = 1;
recon_img = recon_img ./ count_map;

% === 保存结果 ===
save(cfg.output_path, 'recon_img');

% === 保存指标 ===
psnr_val = psnr(recon_img, img2d);
ssim_val = ssim(recon_img, img2d);
metrics.psnr = psnr_val;
metrics.ssim = ssim_val;

fid = fopen(cfg.metrics_path, 'w');
fwrite(fid, jsonencode(metrics), 'char');
fclose(fid);

fprintf('[DONE] PSNR = %.2f dB, SSIM = %.4f\n', psnr_val, ssim_val);
