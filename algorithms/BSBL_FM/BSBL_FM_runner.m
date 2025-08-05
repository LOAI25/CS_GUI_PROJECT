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
patch_size  = cfg.patch_size;   % Patch 大小
stride      = cfg.stride;       % Patch 移动步长（重叠控制）
SNR         = cfg.snr;          % 信噪比
blkLen      = cfg.blk_len;      % BSBL 内部块长度
LearnLambda = cfg.learn_lambda; % BSBL Lambda 学习方式

% === 加载图像数据 ===
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
global_mask = logical(mask_data.mask);  % 转为逻辑型

[H, W] = size(img2d);
fprintf('H = %d, W = %d\n', H, W);

recon_img = zeros(H, W);
count_map = zeros(H, W);

% === DCT 基矩阵 ===
N         = patch_size^2;
blkStartLoc = 1:blkLen:N;
D         = dctmtx(patch_size);
Psi2D     = kron(D', D');

fprintf("=== BSBL-FM Reconstruction with stride=%d, patch_size=%d ===\n", stride, patch_size);

% === 主循环（支持重叠 patch） ===
for i = 1:stride:(H - patch_size + 1)
    for j = 1:stride:(W - patch_size + 1)
        row_range = i:(i + patch_size - 1);
        col_range = j:(j + patch_size - 1);

        % === 提取 patch 图像 & 对应 mask ===
        patch = img2d(row_range, col_range);
        patch_mask = global_mask(row_range, col_range);

        % 将 mask 展平成向量，找到采样点索引
        sample_idx = find(patch_mask(:));

        % 若采样点过少则跳过
        if numel(sample_idx) < 5
            continue;
        end

        % === 构造采样矩阵 Φ ===
        Phi = eye(N);
        Phi = Phi(sample_idx, :);

        % === 压缩感知测量 ===
        x = patch(:);
        y_clean = Phi * x;
        noise_std = std(y_clean) * 10^(-SNR / 20);
        y = y_clean + noise_std * randn(size(y_clean));

        % === 构造 A 矩阵 ===
        A = Phi * Psi2D;

        % === 调用 BSBL-FM 重建 ===
        Result = BSBL_FM(A, y, blkStartLoc, LearnLambda, ...
            'learntype', 0, 'max_iters', 500, 'epsilon', 1e-7, 'verbose', 0);

        theta_recon = Result.x;
        x_recon = Psi2D * theta_recon;
        patch_recon = reshape(x_recon, patch_size, patch_size);
        patch_recon = min(max(patch_recon, 0), 1);  % clip 到 [0, 1]

        % === 累加重建结果 ===
        recon_img(row_range, col_range) = recon_img(row_range, col_range) + patch_recon;
        count_map(row_range, col_range) = count_map(row_range, col_range) + 1;
    end
end

% === 归一化重叠区域 ===
count_map(count_map == 0) = 1;
recon_img = recon_img ./ count_map;

% === 保存重建结果 ===
save(cfg.output_path, 'recon_img');

% === 计算并保存指标 ===
psnr_val = psnr(recon_img, img2d);
ssim_val = ssim(recon_img, img2d);
metrics.psnr = psnr_val;
metrics.ssim = ssim_val;

fid = fopen(cfg.metrics_path, 'w');
fwrite(fid, jsonencode(metrics), 'char');
fclose(fid);

