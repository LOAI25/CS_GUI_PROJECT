function r = r_factor_masked(reconstruction, original, mask, epsval)
%   R-factor for masked (sampled) measurements.
%   r = r_factor_masked(reconstruction, original, mask) computes
%       O = original      .* mask
%       C = reconstruction .* mask
%       R = sum(| |O| - |C| |) / max(sum(|O|), eps)
%
%   Inputs:
%     reconstruction - reconstructed image (real-valued, same size as original)
%     original       - initial image (real-valued)
%     mask           - logical or numeric mask
%     epsval         - (optional) small constant to avoid division by zero (default: eps)
%
%   Output:
%     r              - scalar R-factor (lower is better)

    if nargin < 4 || isempty(epsval), epsval = eps; end

    % Basic validation
    assert(isequal(size(reconstruction), size(original)), ...
        'reconstruction and original must have the same size.');
    assert(isequal(size(mask), size(original)), ...
        'mask must have the same size as images.');

    % Cast to double
    reconstruction = double(reconstruction);
    original       = double(original);
    if islogical(mask)
        mask = double(mask);
    else
        mask = double(mask ~= 0);  % binarize if numeric
    end

    % Sample into observation domain
    O = original       .* mask;
    C = reconstruction .* mask;

    % Apply abs before subtraction
    Oa = abs(O);
    Ca = abs(C);

    % R-factor
    num = sum(abs(Oa(:) - Ca(:)));
    den = sum(Oa(:));
    if den <= epsval, den = epsval; end
    r = num / den;
end
