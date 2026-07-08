function ensure_stage_minus1_fallbacks()
needsFallback = ...
    exist('padarray', 'file') == 0 && exist('padarray', 'builtin') == 0;
needsFallback = needsFallback || ...
    (exist('pdist2', 'file') == 0 && exist('pdist2', 'builtin') == 0);
needsFallback = needsFallback || ...
    (exist('normrnd', 'file') == 0 && exist('normrnd', 'builtin') == 0);
needsFallback = needsFallback || ...
    (exist('fspecial', 'file') == 0 && exist('fspecial', 'builtin') == 0);
needsFallback = needsFallback || ...
    (exist('random', 'file') == 0 && exist('random', 'builtin') == 0);
needsFallback = needsFallback || ...
    (exist('sgolayfilt', 'file') == 0 && exist('sgolayfilt', 'builtin') == 0);

if ~needsFallback
    return;
end

fallbackDir = fullfile(fileparts(mfilename('fullpath')), 'fallbacks');
addpath(fallbackDir, '-end');
fprintf('Stage -1 fallbacks enabled: %s\n', fallbackDir);
end
