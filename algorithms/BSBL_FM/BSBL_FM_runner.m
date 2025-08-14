clear; clc;

% === 切换到项目根目录 ===
cd(fullfile('..', '..'));
addpath(fileparts(mfilename('fullpath')));

% === 读取 config.json ===
fid = fopen('config.json');
if fid == -1
    error("config.json can't be opened");
end
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
cfg = jsondecode(str);

% === 读取配置参数 ===
patch_size  = cfg.patch_size;         % Patch 大小
stride      = cfg.stride;             % Patch 移动步长
blkLen      = cfg.blk_len;            % 块长度
LearnLambda = cfg.learn_lambda;       % 是否学习 lambda
max_iters   = cfg.max_iters;          % 最大迭代次数
epsilon     = cfg.epsilon;            % 收敛精度
learntype   = cfg.learntype;          % 学习类型

% === 加载图像数据 ===
if ~isfile('temp_input.mat')
    error("Can't find temp_input.mat");
end
S = load('temp_input.mat');
img2d = S.input_image;
ref_img = img2d;

% === 全图添加噪声 ===
if isfield(cfg, 'snr') && ~isempty(cfg.snr)
    sigma = std(img2d(:)) * 10^(-cfg.snr / 20);
    img2d = img2d + sigma * randn(size(img2d));
    img2d = min(max(img2d, 0), 1);  % 保证仍在 [0, 1]
    fprintf('[INFO] Added global noise with SNR = %.1f dB\n', cfg.snr);
else
    fprintf('[INFO] No noise added (clean input)\n');
end

% === 加载采样 mask ===
if ~isfile('sampling_mask.mat')
    error("Can't find sampling_mask.mat");
end
mask_data = load('sampling_mask.mat');
if ~isfield(mask_data, 'mask')
    error("sampling_mask.mat does not contain variable 'mask'");
end
global_mask = logical(mask_data.mask);

% === 初始化 ===
[H, W] = size(img2d);
fprintf('H = %d, W = %d\n', H, W);
recon_img = zeros(H, W);
count_map = zeros(H, W);

% === DCT 基 ===
N = patch_size^2;
blkStartLoc = 1:blkLen:N;
D = dctmtx(patch_size);
Psi2D = kron(D', D');

fprintf("=== BSBL-FM Reconstruction with stride=%d, patch_size=%d ===\n", stride, patch_size);

% === 重建主循环 ===
for i = 1:stride:(H - patch_size + 1)
    for j = 1:stride:(W - patch_size + 1)
        row_range = i:(i + patch_size - 1);
        col_range = j:(j + patch_size - 1);

        patch = img2d(row_range, col_range);
        patch_mask = global_mask(row_range, col_range);
        sample_idx = find(patch_mask(:));

        if numel(sample_idx) < 5
            continue;
        end

        Phi = eye(N);
        Phi = Phi(sample_idx, :);

        x = patch(:);
        y = Phi * x;

        A = Phi * Psi2D;

        % === BSBL-FM 重建 ===
        Result = BSBL_FM(A, y, blkStartLoc, LearnLambda, ...
            'learntype', learntype, ...
            'max_iters', max_iters, ...
            'epsilon', epsilon, ...
            'verbose', 0);

        theta_recon = Result.x;
        x_recon = Psi2D * theta_recon;
        patch_recon = reshape(x_recon, patch_size, patch_size);
        patch_recon = min(max(patch_recon, 0), 1);  % 限制在 [0, 1]

        recon_img(row_range, col_range) = recon_img(row_range, col_range) + patch_recon;
        count_map(row_range, col_range) = count_map(row_range, col_range) + 1;
    end
end

% === 重叠区域归一化 ===
count_map(count_map == 0) = 1;
recon_img = recon_img ./ count_map;

% === 保存结果 ===
save(cfg.output_path, 'recon_img');

% === 计算评价指标 ===
psnr_val = psnr(recon_img, ref_img);
ssim_val = ssim(recon_img, ref_img);
metrics.psnr = psnr_val;
metrics.ssim = ssim_val;

fid = fopen(cfg.metrics_path, 'w');
fwrite(fid, jsonencode(metrics), 'char');
fclose(fid);
