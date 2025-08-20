clear; clc;
cd(fullfile('..', '..'));
addpath(fileparts(mfilename('fullpath')));
addpath(fullfile('algorithms','utils'));

fid = fopen('config.json');
if fid == -1
    error("config.json can't be opened");
end
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
cfg = jsondecode(str);

patch_size  = cfg.patch_size; 
stride      = cfg.stride; 
blkLen      = cfg.blk_len;   
LearnLambda = cfg.learn_lambda; 
max_iters   = cfg.max_iters;  
epsilon     = cfg.epsilon; 
learntype   = cfg.learntype;  

if ~isfile('temp_input.mat')
    error("Can't find temp_input.mat");
end
S = load('temp_input.mat');
img2d = S.input_image;
ref_img = img2d;

if isfield(cfg, 'snr') && ~isempty(cfg.snr)
    sigma = std(img2d(:)) * 10^(-cfg.snr / 20);
    img2d = img2d + sigma * randn(size(img2d));
    img2d = min(max(img2d, 0), 1);
end

if ~isfile('sampling_mask.mat')
    error("Can't find sampling_mask.mat");
end
mask_data = load('sampling_mask.mat');
if ~isfield(mask_data, 'mask')
    error("sampling_mask.mat does not contain variable 'mask'");
end
global_mask = logical(mask_data.mask);

[H, W] = size(img2d);
recon_img = zeros(H, W);
count_map = zeros(H, W);

N = patch_size^2;
blkStartLoc = 1:blkLen:N;
D = dctmtx(patch_size);
Psi2D = kron(D', D');

for i = 1:stride:(H - patch_size + 1)
    for j = 1:stride:(W - patch_size + 1)
        row_range = i:(i + patch_size - 1);
        col_range = j:(j + patch_size - 1);

        patch = img2d(row_range, col_range);
        patch_mask = global_mask(row_range, col_range);
        sample_idx = find(patch_mask(:));

        Phi = eye(N);
        Phi = Phi(sample_idx, :);

        x = patch(:);
        y = Phi * x;

        A = Phi * Psi2D;

        Result = BSBL_FM(A, y, blkStartLoc, LearnLambda, ...
            'learntype', learntype, ...
            'max_iters', max_iters, ...
            'epsilon', epsilon, ...
            'verbose', 0);

        theta_recon = Result.x;
        x_recon = Psi2D * theta_recon;
        patch_recon = reshape(x_recon, patch_size, patch_size);
        patch_recon = min(max(patch_recon, 0), 1);

        recon_img(row_range, col_range) = recon_img(row_range, col_range) + patch_recon;
        count_map(row_range, col_range) = count_map(row_range, col_range) + 1;
    end
end

count_map(count_map == 0) = 1;
recon_img = recon_img ./ count_map;

save(cfg.output_path, 'recon_img');

psnr_val = psnr(recon_img, ref_img);
ssim_val = ssim(recon_img, ref_img);
r_val = r_factor_masked(recon_img, ref_img, global_mask);
metrics.psnr = psnr_val;
metrics.ssim = ssim_val;
metrics.r_factor = r_val;

fid = fopen(cfg.metrics_path, 'w');
fwrite(fid, jsonencode(metrics), 'char');
fclose(fid);
