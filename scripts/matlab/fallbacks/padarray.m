function output = padarray(input, padSize, padValue, direction)
if nargin < 3
    padValue = 0;
end
if nargin < 4
    direction = 'both';
end
if ~(isnumeric(padValue) || islogical(padValue)) || ~isscalar(padValue)
    error('retinaSNN:PadarrayFallback', ...
        'Stage -1 padarray fallback only supports scalar constant padding.');
end

padSize = double(padSize(:)');
if any(~isfinite(padSize)) || any(padSize < 0) || any(mod(padSize, 1) ~= 0)
    error('retinaSNN:PadarrayFallback', 'padSize must contain nonnegative integers.');
end

rank = max(ndims(input), numel(padSize));
padSize(end + 1:rank) = 0;
inputSize = size(input);
inputSize(end + 1:rank) = 1;

switch lower(char(direction))
    case 'pre'
        prePad = padSize;
        postPad = zeros(1, rank);
    case 'post'
        prePad = zeros(1, rank);
        postPad = padSize;
    case 'both'
        prePad = padSize;
        postPad = padSize;
    otherwise
        error('retinaSNN:PadarrayFallback', ...
            'Unsupported padarray direction: %s.', direction);
end

outputSize = inputSize + prePad + postPad;
output = repmat(cast(padValue, class(input)), outputSize);
subs = cell(1, rank);
for dim = 1:rank
    subs{dim} = (prePad(dim) + 1):(prePad(dim) + inputSize(dim));
end
output(subs{:}) = input;
end
