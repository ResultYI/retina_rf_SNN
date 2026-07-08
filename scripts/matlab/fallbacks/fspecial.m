function kernel = fspecial(kind, hsize, sigma)
if ~strcmpi(kind, 'gaussian')
    error('retinaSNN:FspecialFallback', ...
        'Stage -1 fspecial fallback only supports gaussian.');
end
if nargin < 2 || isempty(hsize)
    hsize = [3 3];
end
if nargin < 3 || isempty(sigma)
    sigma = 0.5;
end
if isscalar(hsize)
    hsize = [hsize hsize];
end

rows = double(hsize(1));
cols = double(hsize(2));
if any(~isfinite([rows cols sigma])) || rows < 1 || cols < 1 || sigma <= 0
    error('retinaSNN:FspecialFallback', 'Invalid gaussian kernel parameters.');
end

[x, y] = meshgrid(1:cols, 1:rows);
x = x - (cols + 1) / 2;
y = y - (rows + 1) / 2;
kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
kernel = kernel / sum(kernel(:));
end
