function output = padarray(input, padSize, padValue, direction)
%PADARRAY Minimal constant-padding fallback when Image Processing Toolbox is absent.

if nargin < 3
    padValue = 0;
end
if nargin < 4
    direction = 'both';
end
if ~isnumeric(padValue) || ~isscalar(padValue)
    error('retinaSNN:UnsupportedPadding', ...
        'The local padarray fallback supports numeric constant padding only.');
end

dimensions = max(ndims(input), numel(padSize));
inputSize = size(input);
inputSize(end + 1:dimensions) = 1;
padSize(end + 1:dimensions) = 0;

switch lower(direction)
    case 'pre'
        before = padSize;
        after = zeros(size(padSize));
    case 'post'
        before = zeros(size(padSize));
        after = padSize;
    case 'both'
        before = padSize;
        after = padSize;
    otherwise
        error('retinaSNN:UnsupportedPaddingDirection', ...
            'Unsupported padding direction: %s', direction);
end

% ponytail: keep this fallback constant-only; install Image Processing Toolbox
% if a future pipeline needs replicate, symmetric, or circular padding.
output = repmat(cast(padValue, 'like', input), inputSize + before + after);
indices = arrayfun(@(n) before(n) + (1:inputSize(n)), ...
    1:dimensions, 'UniformOutput', false);
output(indices{:}) = input;
end
