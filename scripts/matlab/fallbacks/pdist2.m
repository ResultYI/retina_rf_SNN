function [distances, indices] = pdist2(x, y, varargin)
if size(x, 2) ~= size(y, 2)
    error('retinaSNN:Pdist2Fallback', 'Input point dimensions must match.');
end

[mode, k] = parse_args(size(x, 1), varargin{:});
x = double(x);
y = double(y);
distanceMatrix = sqrt(max(0, sum(x.^2, 2) + sum(y.^2, 2)' - 2 * (x * y')));

if strcmp(mode, 'all')
    distances = distanceMatrix;
    indices = [];
    return;
end

[sortedDistances, sortedIndices] = sort(distanceMatrix, 1, 'ascend');
distances = sortedDistances(1:k, :);
indices = sortedIndices(1:k, :);
end

function [mode, k] = parse_args(rowCount, varargin)
mode = 'all';
k = rowCount;
args = varargin;
if ~isempty(args) && strcmpi(char(args{1}), 'euclidean')
    args = args(2:end);
end
if isempty(args)
    return;
end
if numel(args) ~= 2 || ~strcmpi(char(args{1}), 'smallest')
    error('retinaSNN:Pdist2Fallback', ...
        'Stage -1 pdist2 fallback only supports euclidean Smallest K.');
end
k = min(rowCount, double(args{2}));
if ~isfinite(k) || k < 1 || mod(k, 1) ~= 0
    error('retinaSNN:Pdist2Fallback', 'Smallest K must be a positive integer.');
end
mode = 'smallest';
end
