function setup_isetbio_paths()
%SETUP_ISETBIO_PATHS Configure and validate the local ISETBio installation.

isetbioRoot = strtrim(getenv('ISETBIO_ROOT'));
isetcamRoot = strtrim(getenv('ISETCAM_ROOT'));

if isempty(isetbioRoot) || ~isfolder(isetbioRoot)
    error('retinaSNN:MissingISETBio', ...
        'Set ISETBIO_ROOT to a readable ISETBio checkout.');
end
if isempty(isetcamRoot) || ~isfolder(isetcamRoot)
    error('retinaSNN:MissingISETCam', ...
        'Set ISETCAM_ROOT to a readable ISETCam checkout.');
end

addpath(genpath(isetcamRoot));
addpath(genpath(isetbioRoot));

requiredFunctions = {'sceneFromFile', 'oiCreate', 'oiCompute'};
for idx = 1:numel(requiredFunctions)
    if exist(requiredFunctions{idx}, 'file') == 0
        error('retinaSNN:MissingISETFunction', ...
            'Required function is unavailable: %s', requiredFunctions{idx});
    end
end
if exist('cMosaic', 'class') ~= 8
    error('retinaSNN:MissingCMosaic', ...
        'The installed ISETBio checkout does not expose the cMosaic class.');
end

fprintf('ISETBio and ISETCam paths validated.\n');
end
