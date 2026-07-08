function generate_cone_h5_from_images(inputPath, outputPath, configPath, randomSeed)
arguments
    inputPath (1, :) char
    outputPath (1, :) char
    configPath (1, :) char
    randomSeed (1, 1) double {mustBeInteger, mustBeNonnegative}
end

rng(randomSeed, 'twister');
check_isetbio_env();
cfg = read_flat_yaml(configPath);

timeSteps = cfg_int(cfg, 'time_steps', 16);
dtMs = cfg_number(cfg, 'dt_ms', 5.0);
integrationTimeSeconds = dtMs / 1000.0;
fieldOfViewDeg = cfg_number(cfg, 'field_of_view_deg', 0.5);
eccentricityDegs = [cfg_number(cfg, 'eccentricity_x_deg', 0.0) ...
    cfg_number(cfg, 'eccentricity_y_deg', 0.0)];
meanLuminanceCdM2 = cfg_number(cfg, 'mean_luminance_cd_m2', 100.0);
viewingDistanceMeters = cfg_number(cfg, 'viewing_distance_m', 1.0);
imageSizePx = cfg_int(cfg, 'image_size_px', 128);
achromaticStimulus = cfg_bool(cfg, 'achromatic_stimulus_enabled', true);
eyeMovementEnabled = cfg_bool(cfg, 'eye_movement_enabled', true);
displayFile = cfg_text(cfg, 'display_file', 'LCD-Apple.mat');
waveNm = 400:5:700;

[frames, inputKind] = load_input_frames(inputPath, timeSteps, imageSizePx, ...
    achromaticStimulus);
if eyeMovementEnabled && numel(frames) ~= 1
    error('retinaSNN:EyeMovementSequenceUnsupported', ...
        'Eye movement is supported only for a single still image in Stage -1.');
end

oi = compute_optical_image(frames{1}, inputPath, waveNm, displayFile, ...
    fieldOfViewDeg, meanLuminanceCdM2, viewingDistanceMeters);
cm = build_mosaic(oi, integrationTimeSeconds, fieldOfViewDeg, eccentricityDegs);

if eyeMovementEnabled
    [coneResponse, timeAxisSeconds, eyeTraceDegs] = compute_with_eye_movement( ...
        cm, oi, timeSteps, integrationTimeSeconds, randomSeed);
else
    if numel(frames) == 1
        frames = repmat(frames, 1, timeSteps);
        inputKind = 'image_repeated';
    end
    [coneResponse, timeAxisSeconds, eyeTraceDegs] = compute_frame_sequence( ...
        cm, frames, inputPath, waveNm, displayFile, fieldOfViewDeg, ...
        meanLuminanceCdM2, viewingDistanceMeters, integrationTimeSeconds, ...
        randomSeed);
end

conePositionsDegs = single(cm.coneRFpositionsDegs);
coneTypes = uint8(cm.coneTypes(:));
validate_response(coneResponse, conePositionsDegs, coneTypes, timeAxisSeconds);
lmsResponse = build_lms_response(single(coneResponse), coneTypes);
achromaticResponse = single(sum(lmsResponse, 3));
eyeTraceDegs = single(reshape(eyeTraceDegs, numel(timeAxisSeconds), 2));

prepare_output(outputPath);
write_numeric(outputPath, '/cone_response_lms', lmsResponse);
write_numeric(outputPath, '/cone_response_achromatic', achromaticResponse);
write_numeric(outputPath, '/time_axis_seconds', double(timeAxisSeconds(:)));
write_numeric(outputPath, '/cone_xy_deg', conePositionsDegs);
write_numeric(outputPath, '/cone_type', coneTypes);
write_numeric(outputPath, '/eye_movement_xy_deg', eyeTraceDegs);
write_text(outputPath, '/config_json', jsonencode(cfg));
write_text(outputPath, '/source_image_path', inputPath);
write_text(outputPath, '/source_image_id', source_id(inputPath));

write_numeric(outputPath, '/cone_response', achromaticResponse);
write_numeric(outputPath, '/cone_positions_degs', conePositionsDegs);
write_numeric(outputPath, '/cone_types', coneTypes);
write_numeric(outputPath, '/eye_trace_degs', eyeTraceDegs);
write_numeric(outputPath, '/response_shape_time_cone', int64(size(achromaticResponse)));
write_text(outputPath, '/format_version', 'retina-snn-cone-response-v1');
write_text(outputPath, '/signal_name', 'cone_response_achromatic');
write_text(outputPath, '/response_axis_order', 'TIME_CONE');
write_text(outputPath, '/response_units', 'isomerizations_per_integration_time');
write_text(outputPath, '/input_path', inputPath);
write_text(outputPath, '/input_kind', inputKind);

write_metadata(outputPath, cfg, dtMs, fieldOfViewDeg, eccentricityDegs, ...
    randomSeed, achromaticStimulus);
fprintf('Generated Stage -1 cone HDF5: %s\n', outputPath);
fprintf('  logical cone_response_lms [T,Ncone,3] = [%d,%d,3]\n', ...
    size(lmsResponse, 1), size(lmsResponse, 2));
end

function cm = build_mosaic(oi, integrationTimeSeconds, fieldOfViewDeg, eccentricityDegs)
params = cMosaicParams;
params.integrationTime = integrationTimeSeconds;
params.eccentricityDegs = eccentricityDegs;
params.sizeDegs = [fieldOfViewDeg fieldOfViewDeg];
params.micronsPerDegree = oiGet(oi, 'distance per degree', 'um');
cm = cMosaic(params);
cm.noiseFlag = 'none';
end

function [response, timeAxis, eyeTrace] = compute_with_eye_movement( ...
    cm, oi, timeSteps, integrationTimeSeconds, randomSeed)
cm.emGenSequence(timeSteps * integrationTimeSeconds, ...
    'microsaccadeType', 'none', ...
    'centerPaths', true, ...
    'nTrials', 1, ...
    'randomSeed', randomSeed);
[excitation, ~, ~, ~, computedTimeAxis] = cm.compute(oi, ...
    'withFixationalEyeMovements', true, ...
    'nTrials', 1, ...
    'seed', randomSeed);
response = reshape(single(excitation(1, :, :)), timeSteps, []);
timeAxis = double(computedTimeAxis(:));
eyeTrace = squeeze(cm.fixEMobj.emPosArcMin(1, :, :)) / 60;
end

function [response, timeAxis, eyeTrace] = compute_frame_sequence( ...
    cm, frames, inputPath, waveNm, displayFile, fieldOfViewDeg, ...
    meanLuminanceCdM2, viewingDistanceMeters, integrationTimeSeconds, randomSeed)
timeSteps = numel(frames);
response = zeros(timeSteps, numel(cm.coneTypes), 'single');
for frameIndex = 1:timeSteps
    oi = compute_optical_image(frames{frameIndex}, inputPath, waveNm, ...
        displayFile, fieldOfViewDeg, meanLuminanceCdM2, viewingDistanceMeters);
    excitation = cm.compute(oi, 'seed', randomSeed);
    response(frameIndex, :) = reshape(single(excitation(1, :, :)), 1, []);
end
timeAxis = (0:(timeSteps - 1))' * integrationTimeSeconds;
eyeTrace = zeros(timeSteps, 2);
end

function oi = compute_optical_image(inputImage, inputPath, waveNm, displayFile, ...
    fieldOfViewDeg, meanLuminanceCdM2, viewingDistanceMeters)
scene = sceneFromFile(inputImage, 'rgb', meanLuminanceCdM2, displayFile, waveNm);
scene = sceneSet(scene, 'name', sprintf('Retina SNN input: %s', inputPath));
scene = sceneSet(scene, 'distance', viewingDistanceMeters);
scene = sceneSet(scene, 'fov', fieldOfViewDeg);
scene = sceneSet(scene, 'mean luminance', meanLuminanceCdM2);
oi = oiCreate('human');
oi = oiSet(oi, 'wave', waveNm);
oi = oiCompute(oi, scene, 'pad value', 'mean', 'crop', true);
end

function [frames, inputKind] = load_input_frames(inputPath, maxFrames, imageSizePx, achromatic)
imageExts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'};
if isfolder(inputPath)
    files = list_images(inputPath, imageExts);
    frameCount = min(maxFrames, numel(files));
    frames = cell(1, frameCount);
    for frameIndex = 1:frameCount
        frames{frameIndex} = read_frame(fullfile(files(frameIndex).folder, ...
            files(frameIndex).name), imageSizePx, achromatic);
    end
    inputKind = 'frame_directory';
    return;
end
if ~isfile(inputPath)
    error('retinaSNN:InputNotFound', 'Input does not exist: %s', inputPath);
end
frames = {read_frame(inputPath, imageSizePx, achromatic)};
inputKind = 'image';
end

function files = list_images(folder, imageExts)
files = [];
for extIndex = 1:numel(imageExts)
    files = [files; dir(fullfile(folder, ['*' imageExts{extIndex}]))]; %#ok<AGROW>
end
if isempty(files)
    error('retinaSNN:InputNotFound', 'No image frames found in: %s', folder);
end
[~, order] = sort({files.name});
files = files(order);
end

function frame = read_frame(path, imageSizePx, achromatic)
frame = im2double(imread(path));
if ismatrix(frame)
    frame = repmat(frame, 1, 1, 3);
elseif size(frame, 3) == 4
    frame = frame(:, :, 1:3);
end
if size(frame, 3) ~= 3
    error('retinaSNN:InvalidImage', 'Expected RGB-like image: %s', path);
end
if achromatic
    gray = 0.2126 * frame(:, :, 1) + 0.7152 * frame(:, :, 2) + ...
        0.0722 * frame(:, :, 3);
    frame = repmat(gray, 1, 1, 3);
end
frame = resize_nearest(frame, imageSizePx);
end

function resized = resize_nearest(frame, imageSizePx)
if imageSizePx <= 0
    resized = frame;
    return;
end
rows = max(1, round(linspace(1, size(frame, 1), imageSizePx)));
cols = max(1, round(linspace(1, size(frame, 2), imageSizePx)));
resized = frame(rows, cols, :);
end

function lms = build_lms_response(response, coneTypes)
channel = cone_type_channels(coneTypes);
lms = zeros(size(response, 1), size(response, 2), 3, 'single');
for coneIndex = 1:numel(coneTypes)
    lms(:, coneIndex, channel(coneIndex)) = response(:, coneIndex);
end
end

function channel = cone_type_channels(coneTypes)
types = double(coneTypes(:));
uniqueTypes = unique(types);
if all(ismember(uniqueTypes, [2 3 4]))
    channel = types - 1;
elseif all(ismember(uniqueTypes, [1 2 3]))
    channel = types;
else
    error('retinaSNN:UnsupportedConeTypes', ...
        'Expected cone type ids in [2,3,4] or [1,2,3], got %s.', ...
        mat2str(uniqueTypes'));
end
end

function validate_response(response, positions, coneTypes, timeAxis)
if any(~isfinite(response), 'all') || any(response(:) < 0)
    error('retinaSNN:InvalidConeResponse', ...
        'Cone response must be finite and non-negative.');
end
if size(response, 2) ~= size(positions, 1) || size(response, 2) ~= numel(coneTypes)
    error('retinaSNN:ConeCountMismatch', 'Cone count metadata mismatch.');
end
if numel(timeAxis) ~= size(response, 1) || any(diff(timeAxis) <= 0)
    error('retinaSNN:TimeAxisMismatch', 'Invalid time_axis_seconds.');
end
end

function cfg = read_flat_yaml(configPath)
cfg = struct;
lines = splitlines(string(fileread(configPath)));
for idx = 1:numel(lines)
    line = strip(lines(idx));
    if strlength(line) == 0 || startsWith(line, '#')
        continue;
    end
    parts = split(line, ':');
    if numel(parts) < 2
        continue;
    end
    key = char(strip(parts(1)));
    value = strip(join(parts(2:end), ':'));
    cfg.(key) = char(strip_quotes(value));
end
end

function value = strip_quotes(value)
if startsWith(value, '"') && endsWith(value, '"')
    value = extractBetween(value, 2, strlength(value) - 1);
elseif startsWith(value, '''') && endsWith(value, '''')
    value = extractBetween(value, 2, strlength(value) - 1);
end
end

function value = cfg_text(cfg, name, defaultValue)
if isfield(cfg, name)
    value = cfg.(name);
else
    value = defaultValue;
end
end

function value = cfg_number(cfg, name, defaultValue)
value = str2double(cfg_text(cfg, name, num2str(defaultValue)));
if ~isfinite(value)
    error('retinaSNN:InvalidConfig', '%s must be numeric.', name);
end
end

function value = cfg_int(cfg, name, defaultValue)
value = round(cfg_number(cfg, name, defaultValue));
if value <= 0
    error('retinaSNN:InvalidConfig', '%s must be positive.', name);
end
end

function value = cfg_bool(cfg, name, defaultValue)
text = lower(cfg_text(cfg, name, string(defaultValue)));
value = any(strcmp(text, {'true', '1', 'yes'}));
end

function prepare_output(outputPath)
folder = fileparts(outputPath);
if ~isempty(folder) && ~isfolder(folder)
    mkdir(folder);
end
if isfile(outputPath)
    delete(outputPath);
end
end

function write_numeric(path, datasetName, value)
storageValue = hdf5_storage_value(value);
h5create(path, datasetName, hdf5_storage_size(storageValue), ...
    'Datatype', class(value));
h5write(path, datasetName, storageValue);
end

function write_text(path, datasetName, value)
bytes = uint8(unicode2native(char(value), 'UTF-8'));
h5create(path, datasetName, numel(bytes), 'Datatype', 'uint8');
h5write(path, datasetName, reshape(bytes, 1, []));
end

function value = hdf5_storage_value(value)
if isvector(value)
    value = reshape(value, 1, []);
    return;
end
value = permute(value, ndims(value):-1:1);
end

function datasetSize = hdf5_storage_size(value)
if isvector(value)
    datasetSize = numel(value);
    return;
end
datasetSize = size(value);
end

function write_metadata(path, cfg, dtMs, fieldOfViewDeg, eccentricityDegs, ...
    randomSeed, achromaticStimulus)
h5writeatt(path, '/', 'dt_ms', dtMs);
h5writeatt(path, '/', 'field_of_view_deg', fieldOfViewDeg);
h5writeatt(path, '/', 'eccentricity_deg', eccentricityDegs);
h5writeatt(path, '/', 'mosaic_type', 'cMosaic');
h5writeatt(path, '/', 'mosaic_seed', cfg_int(cfg, 'mosaic_seed', randomSeed));
h5writeatt(path, '/', 'stimulus_seed', randomSeed);
h5writeatt(path, '/', 'is_achromatic_stimulus', uint8(achromaticStimulus));
h5writeatt(path, '/', 'achromatic_projection_method', 'type_routed_lms_sum');
h5writeatt(path, '/', 'ISETBio_git_commit', git_commit(getenv('ISETBIO_ROOT')));
h5writeatt(path, '/', 'ISETCam_git_commit', git_commit(getenv('ISETCAM_ROOT')));
h5writeatt(path, '/', 'MATLAB_version', version);
h5writeatt(path, '/', 'generation_date', datestr(now, 30));
end

function commit = git_commit(root)
commit = 'unknown';
if isempty(root) || ~isfolder(root)
    return;
end
[status, text] = system(sprintf('git -C "%s" rev-parse HEAD', root));
if status == 0
    commit = strtrim(text);
end
end

function id = source_id(path)
[~, name, ext] = fileparts(path);
id = [name ext];
end
