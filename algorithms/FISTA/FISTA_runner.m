clear; clc;
cd(fullfile('..', '..'));
addpath(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'FISTA-master')));
addpath(fullfile('algorithms','utils'));


fid = fopen('config.json');
if fid == -1
    error("config.json can't be opened");
end
raw = fread(fid, inf); str = char(raw'); fclose(fid);
cfg = jsondecode(str);

patch_size       = cfg.patch_size;
stride           = cfg.stride;
lambda     = cfg.lambda;
fista_bt     = cfg.fista_backtracking;
L0         = cfg.L0;
eta        = cfg.eta;
max_iters  = cfg.max_iters;
epsilon    = cfg.epsilon;
fista_pos  = cfg.fista_pos;

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


if isfield(cfg, 'snr') && ~isempty(cfg.snr)
    sigma = std(img2d(:)) * 10^(-cfg.snr / 20);
    img2d = img2d + sigma * randn(size(img2d));
    img2d = min(max(img2d, 0), 1); 
end


[H, W] = size(img2d);
N = patch_size^2;
D = dctmtx(patch_size);
Psi2D = kron(D', D'); 

recon_img = zeros(H, W);
count_map = zeros(H, W);

% FISTA options
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

% choose whether use backtracking as a solver
use_bt = opts.backtracking;
solver = [];
if use_bt && exist('fista_lasso_backtracking', 'file')
    solver = @fista_lasso_backtracking;
elseif ~use_bt && exist('fista_lasso', 'file')
    solver = @fista_lasso;
elseif exist('fista_lasso_backtracking', 'file')
    solver = @fista_lasso_backtracking; 
    opts.backtracking = true;
else
    error("Can't find fista_lasso or fista_lasso_backtracking");
end

for i = 1:stride:(H - patch_size + 1)
    for j = 1:stride:(W - patch_size + 1)
        patch = img2d(i:i+patch_size-1, j:j+patch_size-1);
        patch_mask = global_mask(i:i+patch_size-1, j:j+patch_size-1);

        x = patch(:);
        msk = patch_mask(:);

        Phi = eye(N); Phi = Phi(msk, :);
        A = Phi * Psi2D;
        y = Phi * x;

        theta_hat = solver(y, A, zeros(N,1), opts);

        x_hat = Psi2D * theta_hat;
        patch_hat = reshape(x_hat, patch_size, patch_size);
        patch_hat = min(max(patch_hat, 0), 1); 

        recon_img(i:i+patch_size-1, j:j+patch_size-1) = ...
            recon_img(i:i+patch_size-1, j:j+patch_size-1) + patch_hat;
        count_map(i:i+patch_size-1, j:j+patch_size-1) = ...
            count_map(i:i+patch_size-1, j:j+patch_size-1) + 1;
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