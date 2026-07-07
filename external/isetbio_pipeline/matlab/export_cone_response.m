function export_cone_response(inputPath, outputPath, timeSteps, addEyeMovement)
%EXPORT_CONE_RESPONSE Export native ISETBio cone-mosaic movie responses.
%
% The primary output is cone_response with logical axis order [time, cone].
% Values are noise-free cone excitations in isomerizations per integration
% time. Cone positions and types remain separate metadata, preserving the
% irregular mosaic for later local retinal connectivity.

arguments
    inputPath (1, :) char
    outputPath (1, :) char
    timeSteps (1, 1) double {mustBeInteger, mustBePositive} = 8
    addEyeMovement (1, 1) logical = false
end

setup_isetbio_paths();
[inputFrames, inputKind] = load_input_frames(inputPath, timeSteps);
if addEyeMovement && ~strcmp(inputKind, 'image')
    error('retinaSNN:UnsupportedEyeMovementInput', ...
        'Eye-movement export is only supported for still-image input.');
end

waveNm = 400:5:700;
displayFile = env_text('ISETBIO_DISPLAY_FILE', 'LCD-Apple.mat');
fieldOfViewDegs = env_positive_scalar('ISETBIO_FOV_DEGS', 1.0);
meanLuminanceCdM2 = env_positive_scalar('ISETBIO_MEAN_LUMINANCE_CD_M2', 100.0);
viewingDistanceMeters = env_positive_scalar('ISETBIO_VIEWING_DISTANCE_M', 1.0);
integrationTimeSeconds = env_positive_scalar('ISETBIO_INTEGRATION_TIME_S', 0.005);

oi = compute_optical_image(inputFrames{1}, inputPath, waveNm, displayFile, ...
    fieldOfViewDegs, meanLuminanceCdM2, viewingDistanceMeters);
oiSizeDegs = [oiGet(oi, 'w angular') oiGet(oi, 'h angular')];

cmParams = cMosaicParams;
cmParams.integrationTime = integrationTimeSeconds;
cmParams.eccentricityDegs = [0 0];
cmParams.sizeDegs = oiSizeDegs;
cmParams.micronsPerDegree = oiGet(oi, 'distance per degree', 'um');
cm = cMosaic(cmParams);
cm.noiseFlag = 'none';

if addEyeMovement
    requestedTimeAxis = (0:(timeSteps - 1))' * integrationTimeSeconds;
    cm.emGenSequence(timeSteps * integrationTimeSeconds, ...
        'microsaccadeType', 'none', ...
        'centerPaths', true, ...
        'nTrials', 1, ...
        'randomSeed', 0);
    [coneExcitations, ~, ~, ~, computedTimeAxis] = cm.compute(oi, ...
        'withFixationalEyeMovements', true, ...
        'nTrials', 1, ...
        'seed', 0);
    eyeTraceDegs = squeeze(cm.fixEMobj.emPosArcMin(1, :, :)) / 60;
else
    if strcmp(inputKind, 'image')
        inputFrames = repmat(inputFrames, 1, timeSteps);
        inputKind = 'image_repeated';
    end
    timeSteps = numel(inputFrames);
    coneResponse = zeros(timeSteps, numel(cm.coneTypes), 'single');
    for frameIndex = 1:timeSteps
        if frameIndex == 1
            frameOi = oi;
        else
            frameOi = compute_optical_image(inputFrames{frameIndex}, inputPath, ...
                waveNm, displayFile, fieldOfViewDegs, meanLuminanceCdM2, ...
                viewingDistanceMeters);
        end
        [singleExcitation, ~] = cm.compute(frameOi, 'seed', 0);
        coneResponse(frameIndex, :) = reshape(single(singleExcitation(1, :, :)), 1, []);
    end
    computedTimeAxis = (0:(timeSteps - 1))' * integrationTimeSeconds;
    eyeTraceDegs = zeros(timeSteps, 2);
end

if addEyeMovement && size(coneExcitations, 2) ~= timeSteps
    error('retinaSNN:TimeMismatch', ...
        'cMosaic returned %d time samples; expected %d.', ...
        size(coneExcitations, 2), timeSteps);
end

if addEyeMovement
    coneResponse = reshape(single(coneExcitations(1, :, :)), timeSteps, []);
end
conePositionsDegs = single(cm.coneRFpositionsDegs);
coneTypes = uint8(cm.coneTypes(:));
timeAxisSeconds = double(computedTimeAxis(:));
eyeTraceDegs = single(reshape(eyeTraceDegs, timeSteps, 2));

if any(~isfinite(coneResponse), 'all') || any(coneResponse(:) < 0)
    error('retinaSNN:InvalidConeResponse', ...
        'Cone response must be finite and non-negative.');
end
if size(coneResponse, 2) ~= size(conePositionsDegs, 1)
    error('retinaSNN:ConeCountMismatch', ...
        'Response has %d cones, position table has %d.', ...
        size(coneResponse, 2), size(conePositionsDegs, 1));
end

outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~isfolder(outputDir)
    mkdir(outputDir);
end
if isfile(outputPath)
    delete(outputPath);
end

write_numeric(outputPath, '/cone_response', coneResponse);
write_numeric(outputPath, '/cone_positions_degs', conePositionsDegs);
write_numeric(outputPath, '/cone_types', coneTypes);
write_numeric(outputPath, '/time_axis_seconds', timeAxisSeconds);
write_numeric(outputPath, '/eye_trace_degs', eyeTraceDegs);
write_numeric(outputPath, '/response_shape_time_cone', int64(size(coneResponse)));
write_text(outputPath, '/format_version', 'retina-snn-cone-response-v1');
write_text(outputPath, '/signal_name', 'cone_response');
write_text(outputPath, '/response_axis_order', 'TIME_CONE');
write_text(outputPath, '/response_units', 'isomerizations_per_integration_time');
write_text(outputPath, '/input_path', inputPath);
write_text(outputPath, '/input_kind', inputKind);
write_text(outputPath, '/display_file', displayFile);
write_numeric(outputPath, '/integration_time_seconds', integrationTimeSeconds);
write_numeric(outputPath, '/field_of_view_degs', single(oiSizeDegs));
write_numeric(outputPath, '/mean_luminance_cd_m2', meanLuminanceCdM2);
write_numeric(outputPath, '/viewing_distance_m', viewingDistanceMeters);
write_numeric(outputPath, '/eye_movement_enabled', uint8(addEyeMovement));

previewPath = [strip_extension(outputPath) '_preview.png'];
write_preview(previewPath, conePositionsDegs, coneResponse);

fprintf('Exported cone response: %s\n', outputPath);
fprintf('  logical shape [T, Ncone] = [%d, %d]\n', ...
    size(coneResponse, 1), size(coneResponse, 2));
fprintf('  range = [%.6g, %.6g] isomerizations/integration\n', ...
    min(coneResponse, [], 'all'), max(coneResponse, [], 'all'));
fprintf('  preview = %s\n', previewPath);
end

function [frames, inputKind] = load_input_frames(inputPath, maxFrames)
imageExts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'};
if isfolder(inputPath)
    files = [];
    for extIndex = 1:numel(imageExts)
        files = [files; dir(fullfile(inputPath, ['*' imageExts{extIndex}]))]; %#ok<AGROW>
    end
    if isempty(files)
        error('retinaSNN:InputNotFound', 'No image frames found in: %s', inputPath);
    end
    [~, order] = sort({files.name});
    files = files(order);
    frameCount = min(maxFrames, numel(files));
    frames = cell(1, frameCount);
    for frameIndex = 1:frameCount
        framePath = fullfile(files(frameIndex).folder, files(frameIndex).name);
        frames{frameIndex} = read_rgb_frame(framePath);
    end
    inputKind = 'frame_directory';
    return;
end
if ~isfile(inputPath)
    error('retinaSNN:InputNotFound', 'Input does not exist: %s', inputPath);
end
[~, ~, ext] = fileparts(inputPath);
if any(strcmpi(ext, imageExts))
    frames = {read_rgb_frame(inputPath)};
    inputKind = 'image';
    return;
end
reader = VideoReader(inputPath);
frames = {};
while hasFrame(reader) && numel(frames) < maxFrames
    frames{end + 1} = normalize_rgb_frame(readFrame(reader), inputPath); %#ok<AGROW>
end
if isempty(frames)
    error('retinaSNN:InputNotFound', 'No video frames could be read from: %s', inputPath);
end
inputKind = 'video';
end

function frame = read_rgb_frame(path)
frame = normalize_rgb_frame(imread(path), path);
end

function frame = normalize_rgb_frame(rawFrame, source)
frame = im2double(rawFrame);
if ismatrix(frame)
    frame = repmat(frame, 1, 1, 3);
elseif size(frame, 3) == 4
    frame = frame(:, :, 1:3);
end
if size(frame, 3) ~= 3
    error('retinaSNN:InvalidImage', ...
        'Expected grayscale, RGB, or RGBA frame from %s; got %s.', ...
        source, mat2str(size(frame)));
end
end

function oi = compute_optical_image(inputImage, inputPath, waveNm, displayFile, ...
    fieldOfViewDegs, meanLuminanceCdM2, viewingDistanceMeters)
scene = sceneFromFile(inputImage, 'rgb', meanLuminanceCdM2, displayFile, waveNm);
scene = sceneSet(scene, 'name', sprintf('Retina SNN input: %s', inputPath));
scene = sceneSet(scene, 'distance', viewingDistanceMeters);
scene = sceneSet(scene, 'fov', fieldOfViewDegs);
scene = sceneSet(scene, 'mean luminance', meanLuminanceCdM2);
oi = oiCreate('human');
oi = oiSet(oi, 'wave', waveNm);
oi = oiCompute(oi, scene, 'pad value', 'mean', 'crop', true);
end

function write_preview(path, positions, response)
meanResponse = mean(double(response), 1);
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 900 700]);
scatter(double(positions(:, 1)), double(positions(:, 2)), 10, ...
    meanResponse(:), 'filled');
axis equal tight;
set(gca, 'YDir', 'normal');
xlabel('retinal x (deg)');
ylabel('retinal y (deg)');
title('ISETBio cone response (time mean)');
colormap(parula);
colorbar;
exportgraphics(fig, path, 'Resolution', 180);
close(fig);
end

function write_numeric(path, datasetName, value)
h5create(path, datasetName, size(value), 'Datatype', class(value));
h5write(path, datasetName, value);
end

function write_text(path, datasetName, value)
bytes = uint8(unicode2native(char(value), 'UTF-8'));
h5create(path, datasetName, size(bytes), 'Datatype', 'uint8');
h5write(path, datasetName, bytes);
end

function value = env_text(name, defaultValue)
value = strtrim(getenv(name));
if isempty(value)
    value = defaultValue;
end
end

function value = env_positive_scalar(name, defaultValue)
textValue = strtrim(getenv(name));
if isempty(textValue)
    value = defaultValue;
    return;
end
value = str2double(textValue);
if ~isfinite(value) || value <= 0
    error('retinaSNN:InvalidEnvironmentValue', ...
        '%s must be a positive scalar, got "%s".', name, textValue);
end
end

function stem = strip_extension(path)
[folder, name, ~] = fileparts(path);
stem = fullfile(folder, name);
end
