function values = random(distributionName, varargin)
name = lower(char(distributionName));
switch name
    case 'gamma'
        if numel(varargin) ~= 3
            error('retinaSNN:RandomFallback', 'Gamma random requires shape, scale, size.');
        end
        shape = varargin{1};
        scale = varargin{2};
        outputSize = varargin{3};
        values = gamma_random(shape, scale, outputSize);
    case 'normal'
        if numel(varargin) ~= 3
            error('retinaSNN:RandomFallback', 'Normal random requires mu, sigma, size.');
        end
        values = normrnd(varargin{1}, varargin{2}, varargin{3});
    otherwise
        error('retinaSNN:RandomFallback', ...
            'Stage -1 random fallback only supports Gamma and Normal.');
end
end

function values = gamma_random(shape, scale, outputSize)
if ~isscalar(shape) || ~isscalar(scale) || shape <= 0 || scale <= 0
    error('retinaSNN:RandomFallback', ...
        'Gamma fallback only supports positive scalar shape and scale.');
end

count = prod(double(outputSize));
flat = zeros(1, count);
for idx = 1:count
    flat(idx) = scale * gamma_unit(shape);
end
values = reshape(flat, outputSize);
end

function value = gamma_unit(shape)
if shape < 1
    value = gamma_unit(shape + 1) * rand()^(1 / shape);
    return;
end

d = shape - 1 / 3;
c = 1 / sqrt(9 * d);
while true
    x = randn();
    v = (1 + c * x)^3;
    if v <= 0
        continue;
    end
    u = rand();
    if u < 1 - 0.0331 * x^4
        value = d * v;
        return;
    end
    if log(u) < 0.5 * x^2 + d * (1 - v + log(v))
        value = d * v;
        return;
    end
end
end
