function filtered = sgolayfilt(signal, order, frameLength)
if ~isvector(signal)
    error('retinaSNN:SgolayfiltFallback', ...
        'Stage -1 sgolayfilt fallback only supports vectors.');
end
if frameLength <= order || mod(frameLength, 2) == 0
    error('retinaSNN:SgolayfiltFallback', ...
        'frameLength must be odd and greater than order.');
end

wasRow = isrow(signal);
samples = double(signal(:));
halfWidth = (frameLength - 1) / 2;
filtered = zeros(size(samples), 'like', samples);

for index = 1:numel(samples)
    first = max(1, index - halfWidth);
    last = min(numel(samples), index + halfWidth);
    neighbors = (first:last)';
    localOrder = min(order, numel(neighbors) - 1);
    design = ones(numel(neighbors), localOrder + 1);
    offsets = neighbors - index;
    for power = 1:localOrder
        design(:, power + 1) = offsets .^ power;
    end
    coefficients = design \ samples(neighbors);
    filtered(index) = coefficients(1);
end

filtered = cast(filtered, class(signal));
if wasRow
    filtered = filtered.';
end
end
