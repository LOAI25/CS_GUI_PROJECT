clear; clc;
% Path
cd(fullfile('..', '..'));
addpath(fileparts(mfilename('fullpath')));

% read json
fid = fopen('config.json');
if fid == -1
    error("config.json can't be open");
end
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
cfg = jsondecode(str);

% load preprocessed mat file
if ~isfile('temp_input.mat')
    error("Can't find temp_input.mat");
end
S = load('temp_input.mat');
img2d = S.input_image;

% parameters
patch_size     = cfg.block_size;
sampling_rate  = cfg.sampling_rate;
SNR            = cfg.snr;
blkLen         = cfg.blk_len;
LearnLambda    = cfg.learn_lambda;

[H, W] = size(img2d);
recon_img = zeros(H, W);
count_map = zeros(H, W);

N         = patch_size^2;
blkStartLoc = 1:blkLen:N;
M         = round(sampling_rate * N);
D         = dctmtx(patch_size);
Psi2D     = kron(D, D);

% main loop for blocks
for i = 1:patch_size:H
    for j = 1:patch_size:W
        row_range = i:min(i+patch_size-1, H);
        col_range = j:min(j+patch_size-1, W);

        if length(row_range) < patch_size || length(col_range) < patch_size
            continue;
        end

        patch = img2d(row_range, col_range);
        x = patch(:);

        Phi = randn(M, N);
        Phi = Phi ./ vecnorm(Phi);
        y_clean = Phi * x;
        noise_std = std(y_clean) * 10^(-SNR / 20);
        y = y_clean + noise_std * randn(M, 1);
        A = Phi * Psi2D;

        Result = BSBL_FM(A, y, blkStartLoc, LearnLambda, ...
            'learntype', 0, 'max_iters', 500, 'epsilon', 1e-7, 'verbose', 0);

        theta_recon = Result.x;
        x_recon = Psi2D * theta_recon;
        patch_recon = reshape(x_recon, patch_size, patch_size);
        patch_recon = min(max(patch_recon, 0), 1);  % clip

        recon_img(row_range, col_range) = recon_img(row_range, col_range) + patch_recon;
        count_map(row_range, col_range) = count_map(row_range, col_range) + 1;
    end
end

count_map(count_map == 0) = 1;
recon_img = recon_img ./ count_map;

% save recon result
save(cfg.output_path, 'recon_img');

% write metrics
psnr_val = psnr(recon_img, img2d);
ssim_val = ssim(recon_img, img2d);
metrics.psnr = psnr_val;
metrics.ssim = ssim_val;

fid = fopen(cfg.metrics_path, 'w');
fwrite(fid, jsonencode(metrics), 'char');
fclose(fid);


