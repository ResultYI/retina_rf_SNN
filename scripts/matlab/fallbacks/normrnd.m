function values = normrnd(mu, sigma, varargin)
if nargin < 2
    error('retinaSNN:NormrndFallback', 'normrnd requires mu and sigma.');
end
if any(sigma(:) < 0)
    error('retinaSNN:NormrndFallback', 'sigma must be nonnegative.');
end

if isempty(varargin)
    values = mu + sigma .* randn(size(mu + sigma));
else
    values = mu + sigma .* randn(varargin{:});
end
end
