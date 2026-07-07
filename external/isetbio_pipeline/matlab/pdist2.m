function [distances, indices] = pdist2(x, y, metric, option, count)
%PDIST2 Minimal batched Euclidean nearest-neighbor fallback for cMosaic.

if ~strcmpi(metric, 'euclidean') || ~strcmpi(option, 'Smallest')
    error('retinaSNN:UnsupportedPdist2', ...
        'The local pdist2 fallback supports Euclidean Smallest K only.');
end
if size(x, 2) ~= size(y, 2) || count > size(x, 1)
    error('retinaSNN:InvalidPdist2Input', 'Incompatible point arrays or K.');
end

distances = zeros(count, size(y, 1), 'like', x);
indices = zeros(count, size(y, 1));
xSquared = sum(x .* x, 2);

% ponytail: batching avoids an NxN allocation; use the Statistics Toolbox
% implementation if future code needs other metrics or output modes.
batchSize = 256;
for first = 1:batchSize:size(y, 1)
    columns = first:min(first + batchSize - 1, size(y, 1));
    block = y(columns, :);
    squared = max(xSquared + sum(block .* block, 2)' - 2 * (x * block'), 0);
    [smallest, nearest] = mink(squared, count, 1);
    distances(:, columns) = sqrt(smallest);
    indices(:, columns) = nearest;
end
end
