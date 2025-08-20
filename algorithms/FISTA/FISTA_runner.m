clear; clc;

% === 切换到项目根目录 ===
cd(fullfile('..', '..'));
addpath(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'FISTA-master')));
addpath(fullfile('algorithms','utils'));

% === 加载配置 ===
fid = fopen('config.json');
if fid == -1, error("config.json can't be opened"); end
raw = fread(fid, inf); str = char(raw'); fclose(fid);
cfg = jsondecode(str);

% === 参数提取（含 FISTA 相关）===
patch_size       = cfg.patch_size;
stride           = cfg.stride;
lambda     = cfg.lambda;
fista_bt     = cfg.fista_backtracking;
L0         = cfg.L0;
eta        = cfg.eta;
max_iters  = cfg.max_iters;
epsilon    = cfg.epsilon;
fista_pos  = cfg.fista_pos;

% === 加载输入图像 ===
if ~isfile('temp_input.mat')
    error("Can't find temp_input.mat");
end
S = load('temp_input.mat');
img2d = S.input_image;
ref_img = img2d;

% === 加载采样 mask ===
if ~isfile('sampling_mask.mat')
    error("Can't find sampling_mask.mat");
end
mask_data = load('sampling_mask.mat');
if ~isfield(mask_data, 'mask')
    error("sampling_mask.mat does not contain variable 'mask'");
end
global_mask = logical(mask_data.mask);

% === 可选：全图添加噪声（与 GUI 管线一致） ===
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
Psi2D = kron(D', D');   % (N x N)  IDCT 字典：x = Psi2D * alpha

recon_img = zeros(H, W);
count_map = zeros(H, W);

% === FISTA 选项 ===
opts = struct();
opts.lambda       = lambda;
opts.pos          = logical(fista_pos);
opts.backtracking = logical(fista_bt);
opts.max_iters    = max_iters;
opts.epsilon      = epsilon;
opts.check_grad   = false;
if opts.backtracking
    opts.L0 = L0; opts.eta = eta;
end

% 选择求解器（优先用 backtracking 版）
use_bt = opts.backtracking;
solver = [];
if use_bt && exist('fista_lasso_backtracking', 'file')
    solver = @fista_lasso_backtracking;
elseif ~use_bt && exist('fista_lasso', 'file')
    solver = @fista_lasso;
elseif exist('fista_lasso_backtracking', 'file')
    solver = @fista_lasso_backtracking;  % 兜底：可把 backtracking 当普通 FISTA 用
    opts.backtracking = true;
    warning('[WARN] fista_lasso 不存在，已使用 fista_lasso_backtracking。');
else
    error('找不到 fista_lasso 或 fista_lasso_backtracking，请确认路径。');
end

fprintf("=== FISTA Reconstruction with stride=%d, patch_size=%d, lambda=%.3g, BT=%d ===\n", ...
        stride, patch_size, lambda, opts.backtracking);

% === 主循环（重叠分块 + 压缩感知/FISTA）===
for i = 1:stride:(H - patch_size + 1)
    for j = 1:stride:(W - patch_size + 1)
        patch = img2d(i:i+patch_size-1, j:j+patch_size-1);
        patch_mask = global_mask(i:i+patch_size-1, j:j+patch_size-1);

        x = patch(:);
        msk = patch_mask(:);


        Phi = eye(N); Phi = Phi(msk, :);
        A = Phi * Psi2D;
        y = Phi * x;

        % === 调用 SPGL1 ===
        theta_hat = solver(y, A, zeros(N,1), opts);

        % 空间域重建：x_hat = Ψ θ
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
psnr_val = psnr(recon_img, ref_img);
ssim_val = ssim(recon_img, ref_img);
r_val = r_factor_masked(recon_img, ref_img, global_mask);
metrics.psnr = psnr_val;
metrics.ssim = ssim_val;
metrics.r_factor = r_val;

fid = fopen(cfg.metrics_path, 'w');
fwrite(fid, jsonencode(metrics), 'char');
fclose(fid);

fprintf('[DONE] PSNR = %.2f dB, SSIM = %.4f\n', psnr_val, ssim_val);