function report = assessFundusQuality(rawImage, varargin)
% assessFundusQuality: CANONICAL Stage-1 quality API (struct output) and
% SINGLE implementation of the ROI / focus / entropy / enhancement core.
% assessAndEnhanceImage.m is a thin backward-compatible 6-output wrapper
% delegating here (legacy boolean = threshold-only comparison; extended
% FAIL reasons such as degenerate coverage live in report.decision).
% New code must call THIS.
%
% DECISION: PASS | BORDERLINE | FAIL
%   FAIL      - below focusThresh or entropyThresh, or ROI degenerate,
%               or severe exposure/artefact. Do not grade; recapture.
%   BORDERLINE- above thresholds but within configured margins, or mild
%               exposure/contrast warnings. Gradeable-with-warning: Stage 2
%               may proceed but must log warnings; curators should review.
%               NEVER silently coerce to PASS - report.decision stays
%               'BORDERLINE' and report.reasons is non-empty.
%   PASS      - above thresholds + margins, no severe flags.
%
% USAGE:
%   rep = assessFundusQuality(img);
%   rep = assessFundusQuality(img, 'FocusThresh', 9, 'EntropyThresh', 3.8);
%   rep = assessFundusQuality(img, 'Config', qualityConfig(), 'CalibrationDir', pwd);
%   rep = assessFundusQuality(img, 'FocusThresh', -Inf, 'EntropyThresh', -Inf, ...
%             'CalibrationDir', []);  % enhance-only, never rejects
%
% CalibrationDir: folder searched for qualityThresholds.mat ([] = skip
%   calibration entirely and use canonical/explicit thresholds; the legacy
%   wrapper passes [] so explicit caller thresholds are never overridden).
%   Explicit FocusThresh/EntropyThresh always win over the .mat file.
%   BORDERLINE margins are derived from the EFFECTIVE thresholds.
%
% OUTPUT report struct fields:
%   decision, isGradeable (= decision ~= 'FAIL'), legacyGradeable
%     (pure focus>=ft && entropy>=et comparison, exactly what the legacy
%     wrapper returns), reasons{}, recaptureGuidance{}
%   focusScore, entropyScore, focusThresh, entropyThresh (effective)
%   scores struct (illumination, exposure, contrast, artefact, coverage...)
%   enhancedRGB (HxWx3 uint8, bg=0), enhancedGray (HxW uint8, bg=0),
%   roiMask (HxW logical), roiInfo struct
%   configVersion, isCalibrated, calibrationSource
%
% Requires: Image Processing Toolbox. Deterministic (no rng).

p = inputParser;
addParameter(p, 'FocusThresh', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
addParameter(p, 'EntropyThresh', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
addParameter(p, 'Config', qualityConfig(), @isstruct);
addParameter(p, 'CalibrationDir', pwd, @(x) isempty(x) || ischar(x));
parse(p, varargin{:});
cfg = p.Results.Config;
if isempty(cfg), cfg = qualityConfig(); end

if isempty(p.Results.CalibrationDir)
    focusThresh = cfg.focusThresh; entropyThresh = cfg.entropyThresh;
    calibInfo = struct('isCalibrated', false, 'source', 'explicit caller thresholds (calibration bypassed)', 'note', 'legacy-explicit');
else
    [focusThresh, entropyThresh, calibInfo] = qualityLoadCalibration(cfg, p.Results.CalibrationDir);
end
if ~isempty(p.Results.FocusThresh), focusThresh = p.Results.FocusThresh; end
if ~isempty(p.Results.EntropyThresh), entropyThresh = p.Results.EntropyThresh; end

t0 = cputime;

% --- 0. Validate + normalize (identical to legacy core) ---
if isempty(rawImage) || ~isnumeric(rawImage) && ~islogical(rawImage)
    error('assessFundusQuality:badInput', 'rawImage must be a non-empty numeric/logical image.');
end
if ndims(rawImage) ~= 2 && ndims(rawImage) ~= 3
    error('assessFundusQuality:badInput', 'rawImage must be HxW or HxWxC.');
end
if size(rawImage,3) > 3
    warning('assessFundusQuality:extraChannels', ...
        'Got %d channels; keeping first 3 (RGB) and ignoring alpha/depth extras.', size(rawImage,3));
    rawImage = rawImage(:,:,1:3);
end
if size(rawImage,3) == 1
    rawImage = repmat(rawImage, [1 1 3]);
end
rawImage = im2uint8(rawImage);

% --- 1. Fundus-circle ROI (single implementation) ---
grayFull = rgb2gray(rawImage);
roiMask = grayFull > cfg.roiSeedThresh;
roiMask = imfill(roiMask, 'holes');
cc = bwconncomp(roiMask, 8); % explicit 8-connectivity (never rely on defaults)
if cc.NumObjects > 1
    % Largest component wins: dust specks, sticker tags, and light-leak
    % corners are orders of magnitude smaller than the fundus disc.
    sizes = cellfun(@numel, cc.PixelIdxList);
    [~, biggest] = max(sizes);
    roiMask = false(size(roiMask));
    roiMask(cc.PixelIdxList{biggest}) = true;
end
% Edge-shave scales with resolution: floor preserves validated 1280px
% behaviour, large IDRiD frames get proportionally more.
roiDiaEst = sqrt(4 * nnz(roiMask) / pi);
erodeR = max(cfg.roiErodeDisk, round(cfg.roiErodeRelFrac * roiDiaEst));
roiMask = imerode(roiMask, strel('disk', erodeR));
if ~any(roiMask(:))
    roiMask = true(size(grayFull)); % degenerate input - fall back rather than crash
end

% --- 2. Quality scores, restricted to the ROI ---
laplacianFilter = fspecial('laplacian', cfg.laplacianAlpha);
laplacianImage = imfilter(double(grayFull), laplacianFilter, 'replicate');
focusScore = var(laplacianImage(roiMask));

hsvImage = rgb2hsv(rawImage);
vChannel = hsvImage(:,:,3);
entropyScore = localMaskedEntropy(vChannel, roiMask, cfg.entropyBins);

legacyGradeable = (focusScore >= focusThresh) && (entropyScore >= entropyThresh);

% --- 3. Adaptive enhancement (single implementation) ---
roiDiameter = sqrt(4 * nnz(roiMask) / pi);
bgRadius = max(round(cfg.bgFraction * roiDiameter), cfg.bgFloorPx);
structuringElement = strel('disk', bgRadius);

% Guardrail: probe gray ROI contrast BEFORE CLAHE. Flat (noisy/low-light)
% images get halved ClipLimit so CLAHE does not turn sensor noise into
% phantom texture; tiny thumbnails skip non-local-means denoise which
% would erase 1-2px vessels. Normal images take the standard path.
grayROI = double(grayFull(roiMask)) / 255;
flatContrast = prctile(grayROI, 95) - prctile(grayROI, 5);
cfgEnh = cfg;
if flatContrast < cfg.enhanceFlatThresh
    cfgEnh.claheClipLimit = cfg.enhanceFlatClip;
end
cfgEnh.doDenoise = roiDiameter >= cfg.denoiseMinDiameter;

greenChannel = rawImage(:,:,2);
greenBackground = imopen(greenChannel, structuringElement);
% Divide-based flat-fielding (not subtraction): division rescales dim
% periphery instead of crushing it to black.
enhancedGray = localFlatFieldClaheDenoise(greenChannel, greenBackground, roiMask, cfgEnh);
enhancedGray(~roiMask) = 0;

enhancedRGB = rawImage;
for c = 1:3
    chan = rawImage(:,:,c);
    bg = imopen(chan, structuringElement);
    enhancedRGB(:,:,c) = localFlatFieldClaheDenoise(chan, bg, roiMask, cfgEnh);
end
for c = 1:3
    chanMasked = enhancedRGB(:,:,c);
    chanMasked(~roiMask) = 0;
    enhancedRGB(:,:,c) = chanMasked;
end

% --- 4. Extended metrics (ROI-masked, resolution-normalized) ---
V = vChannel;
roi = roiMask;
roiVals = double(V(roi));
framePx = numel(roi);

coverageFrac = nnz(roi) / framePx;
perimMask = bwperim(roi);
perimPx = nnz(perimMask);
if perimPx > 0
    circularity = 4 * pi * nnz(roi) / (perimPx^2);
    circularity = min(circularity, 1);
else
    circularity = 0;
end

illumMedian = median(roiVals);
illumP5  = prctile(roiVals, cfg.illumLowPctl);
illumP95 = prctile(roiVals, cfg.illumHighPctl);
contrastP95P5 = illumP95 - illumP5;
underexpFrac = mean(roiVals < cfg.underexposedV);
overexpFrac  = mean(roiVals > cfg.overexposedV);
specularFrac = mean(roiVals > cfg.specularV);

scores = struct('illuminationMedian', illumMedian, 'illuminationP5', illumP5, ...
    'illuminationP95', illumP95, 'contrastP95P5', contrastP95P5, ...
    'underexposedFrac', underexpFrac, 'overexposedFrac', overexpFrac, ...
    'specularFrac', specularFrac, 'coverageFrac', coverageFrac, ...
    'circularity', circularity, 'roiDiameterPx', roiDiameter, ...
    'focusScore', focusScore, 'entropyScore', entropyScore);

% --- 5. Decision with reasons + guidance (margins from EFFECTIVE thresholds) ---
reasons = {};
guidance = {};

focusBorder = focusThresh * (1 + cfg.borderlineFocusMargin);
entropyBorder = entropyThresh * (1 + cfg.borderlineEntropyMargin);

if coverageFrac < cfg.roiMinCoverageFrac
    reasons{end+1} = sprintf('ROI coverage %.3f below minimum %.3f: no usable fundus FOV detected.', coverageFrac, cfg.roiMinCoverageFrac);
    guidance{end+1} = 'Recapture: center the optic disc/macula, fill the frame with retina, remove lens cap/obstruction.';
end
if circularity < cfg.roiCircularityMin && coverageFrac >= cfg.roiMinCoverageFrac
    reasons{end+1} = sprintf('ROI circularity %.2f below %.2f: FOV is clipped or severely cropped.', circularity, cfg.roiCircularityMin);
    guidance{end+1} = 'Recapture: recenter the eye, keep eyelids/lashes clear of the pupil, avoid heavy cropping.';
end
if ~(focusScore >= focusThresh)
    reasons{end+1} = sprintf('Focus %.2f below threshold %.2f: image is blurred.', focusScore, focusThresh);
    guidance{end+1} = 'Recapture: hold the camera steady 2s, refocus on vessels near the disc, clean the lens.';
elseif focusScore < focusBorder
    reasons{end+1} = sprintf('Focus %.2f within %.0f%% margin above threshold %.2f: borderline sharpness.', focusScore, cfg.borderlineFocusMargin*100, focusThresh);
    guidance{end+1} = 'Caution: acceptable but soft - refocus if a sharper repeat is cheap; flag for curator review.';
end
if ~(entropyScore >= entropyThresh)
    reasons{end+1} = sprintf('Entropy %.2f below threshold %.2f: poor illumination/contrast.', entropyScore, entropyThresh);
    guidance{end+1} = 'Recapture: reduce room light, set correct flash, remove glare, check exposure.';
elseif entropyScore < entropyBorder
    reasons{end+1} = sprintf('Entropy %.2f within %.0f%% margin above threshold %.2f: borderline illumination.', entropyScore, cfg.borderlineEntropyMargin*100, entropyThresh);
    guidance{end+1} = 'Caution: dim/flat lighting - repeat with adjusted flash if possible.';
end
if underexpFrac > 0.25
    reasons{end+1} = sprintf('Underexposed fraction %.2f: more than a quarter of ROI is near-black.', underexpFrac);
    guidance{end+1} = 'Recapture: increase flash/exposure, darken the room to dilate the pupil naturally.';
end
if overexpFrac > 0.15
    reasons{end+1} = sprintf('Overexposed fraction %.2f: widespread saturation washes out lesions.', overexpFrac);
    guidance{end+1} = 'Recapture: lower flash, avoid direct reflections, clean condensation from the lens.';
end
if specularFrac > 0.02
    reasons{end+1} = sprintf('Specular artefact fraction %.3f: strong reflections present.', specularFrac);
    guidance{end+1} = 'Recapture: angle the camera slightly off-axis, ask patient to blink, wipe the lens.';
end
if contrastP95P5 < 0.25
    reasons{end+1} = sprintf('Contrast (P95-P5) %.2f is flat: vessels/lesions will be hard to separate.', contrastP95P5);
    guidance{end+1} = 'Recapture with better focus/flash; enhancement cannot invent missing contrast.';
end

hardFail = ~legacyGradeable || coverageFrac < cfg.roiMinCoverageFrac || underexpFrac > 0.25 || overexpFrac > 0.15;
isBorderline = ~hardFail && (~isempty(reasons));
if hardFail
    decision = 'FAIL';
elseif isBorderline
    decision = 'BORDERLINE';
else
    decision = 'PASS';
end
isGradeable = ~strcmp(decision, 'FAIL');

roiInfo = struct('coverageFrac', coverageFrac, 'circularity', circularity, ...
    'roiDiameterPx', roiDiameter, 'numPixels', nnz(roiMask), 'framePixels', framePx);

report = struct('decision', decision, 'isGradeable', isGradeable, ...
    'legacyGradeable', legacyGradeable, ...
    'reasons', {reasons}, 'recaptureGuidance', {guidance}, ...
    'focusScore', focusScore, 'entropyScore', entropyScore, ...
    'focusThresh', focusThresh, 'entropyThresh', entropyThresh, ...
    'scores', scores, 'enhancedRGB', enhancedRGB, 'enhancedGray', enhancedGray, ...
    'roiMask', roiMask, 'roiInfo', roiInfo, ...
    'configVersion', cfg.version, 'isCalibrated', calibInfo.isCalibrated, ...
    'calibrationSource', calibInfo.source, 'elapsedCpuSec', cputime - t0);
end

% ------------------------------------------------------------------
function out = localFlatFieldClaheDenoise(chan, background, roiMask, cfg)
flatField = double(chan) ./ (double(background) + 1);
normalizer = max(flatField(roiMask));
if normalizer <= 0 || ~isfinite(normalizer)
    normalizer = 1;
end
flatField = min(flatField ./ normalizer, 1);
flatFieldU8 = im2uint8(flatField);
claheChan = adapthisteq(flatFieldU8, 'ClipLimit', cfg.claheClipLimit, 'NumTiles', cfg.claheNumTiles);
if isfield(cfg, 'doDenoise') && ~cfg.doDenoise
    out = claheChan;
else
    out = imnlmfilt(claheChan);
end
end

% ------------------------------------------------------------------
function e = localMaskedEntropy(channel01, mask, nBins)
% Shannon entropy (base 2, nBins-bin histogram), restricted to mask==true.
if nargin < 3 || isempty(nBins), nBins = 256; end
vals = channel01(mask);
counts = histcounts(vals, nBins, 'BinLimits', [0 1]);
p = counts / sum(counts);
p = p(p > 0);
e = -sum(p .* log2(p));
end
