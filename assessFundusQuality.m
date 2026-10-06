function report = assessFundusQuality(rawImage, varargin)
% assessFundusQuality: CANONICAL Stage-1 quality API (struct output).
% assessAndEnhanceImage.m remains as the backward-compatible 6-output
% wrapper (delegating here since v1.2) - new code should call THIS.
%
% DECISION: PASS | BORDERLINE | FAIL
%   FAIL      - below focusThresh or entropyThresh, or ROI degenerate,
%               or severe exposure/artefact. Do not grade; recapture.
%   BORDERLINE- above thresholds but within configured margins, or mild
%               exposure/contrast warnings. Gradeable-with-warning: Stage 2
%               may proceed but must log warnings; curators should review.
%   PASS      - above thresholds + margins, no severe flags.
%
% USAGE:
%   rep = assessFundusQuality(img);
%   rep = assessFundusQuality(img, 'FocusThresh', 9, 'EntropyThresh', 3.8);
%   rep = assessFundusQuality(img, 'Config', qualityConfig());
%
% OUTPUT report struct fields:
%   decision, isGradeable, reasons{}, recaptureGuidance{}
%   focusScore, entropyScore, focusThresh, entropyThresh
%   scores struct (illumination, exposure, contrast, artefact, coverage...)
%   enhancedRGB, enhancedGray, roiMask, roiInfo struct
%   configVersion, isCalibrated, calibrationSource
%
% Requires: Image Processing Toolbox. Deterministic (no rng).

p = inputParser;
addParameter(p, 'FocusThresh', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
addParameter(p, 'EntropyThresh', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
addParameter(p, 'Config', qualityConfig(), @isstruct);
addParameter(p, 'CalibrationDir', pwd, @ischar);
parse(p, varargin{:});
cfg = p.Results.Config;
if isempty(cfg), cfg = qualityConfig(); end

[focusThresh, entropyThresh, calibInfo] = qualityLoadCalibration(cfg, p.Results.CalibrationDir);
if ~isempty(p.Results.FocusThresh), focusThresh = p.Results.FocusThresh; end
if ~isempty(p.Results.EntropyThresh), entropyThresh = p.Results.EntropyThresh; end

t0 = cputime;
[isGradeableBase, enhancedRGB, enhancedGray, focusScore, entropyScore, roiMask] = ...
    assessAndEnhanceImage(rawImage, focusThresh, entropyThresh);

% --- Extended metrics from raw + roi (all ROI-masked, resolution-normalized) ---
if size(rawImage,3) == 1
    rgbU8 = repmat(im2uint8(rawImage), [1 1 3]);
else
    rgbU8 = im2uint8(rawImage);
end
hsv = rgb2hsv(rgbU8);
V = hsv(:,:,3);
roi = roiMask;
if ~any(roi(:))
    roi = true(size(V));
end
roiVals = double(V(roi));
framePx = numel(roi);

coverageFrac = nnz(roi) / framePx;
% Circularity 4*pi*A/P^2 on largest-component boundary
perimMask = bwperim(roi);
perimPx = nnz(perimMask);
if perimPx > 0
    circularity = 4 * pi * nnz(roi) / (perimPx^2);
    circularity = min(circularity, 1);
else
    circularity = 0;
end
roiDiameter = sqrt(4 * nnz(roi) / pi);

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

% --- Decision with reasons + guidance ---
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

hardFail = ~isGradeableBase || coverageFrac < cfg.roiMinCoverageFrac || underexpFrac > 0.25 || overexpFrac > 0.15;
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
    'reasons', {reasons}, 'recaptureGuidance', {guidance}, ...
    'focusScore', focusScore, 'entropyScore', entropyScore, ...
    'focusThresh', focusThresh, 'entropyThresh', entropyThresh, ...
    'scores', scores, 'enhancedRGB', enhancedRGB, 'enhancedGray', enhancedGray, ...
    'roiMask', roiMask, 'roiInfo', roiInfo, ...
    'configVersion', cfg.version, 'isCalibrated', calibInfo.isCalibrated, ...
    'calibrationSource', calibInfo.source, 'elapsedCpuSec', cputime - t0);
end
