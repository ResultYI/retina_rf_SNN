function check_isetbio_env()
add_env_path('ISETCAM_ROOT');
add_env_path('ISETBIO_ROOT');
ensure_stage_minus1_fallbacks();

fprintf('MATLAB version: %s\n', version);
ver;

requiredFunctions = {'ieInit', 'sceneFromFile', 'oiCreate', 'cMosaic', 'compute'};
missing = {};
for idx = 1:numel(requiredFunctions)
    name = requiredFunctions{idx};
    if ~function_available(name)
        missing{end + 1} = name; %#ok<AGROW>
    end
end

if ~isempty(missing)
    error('retinaSNN:ISETBioEnvironment', ...
        'Missing MATLAB/ISETBio symbols: %s. Set ISETBIO_ROOT and ISETCAM_ROOT or add them to the MATLAB path.', ...
        strjoin(missing, ', '));
end

fprintf('ISETBio/ISETCam environment check passed.\n');
fprintf('ISETBIO_ROOT=%s\n', getenv('ISETBIO_ROOT'));
fprintf('ISETCAM_ROOT=%s\n', getenv('ISETCAM_ROOT'));
end

function add_env_path(name)
root = strtrim(getenv(name));
if ~isempty(root) && isfolder(root)
    addpath(genpath(root));
end
end

function ok = function_available(name)
if strcmp(name, 'cMosaic')
    ok = exist('cMosaic', 'class') == 8;
    return;
end
if strcmp(name, 'compute')
    ok = exist('compute', 'file') ~= 0;
    if exist('cMosaic', 'class') == 8
        ok = ok || any(strcmp(methods('cMosaic'), 'compute'));
    end
    return;
end
ok = exist(name, 'file') ~= 0 || exist(name, 'builtin') ~= 0;
end
