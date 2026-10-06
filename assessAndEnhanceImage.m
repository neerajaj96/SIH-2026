function [isGradeable, enhancedRGB, enhancedGray, focusScore, entropyScore, roiMask] = ...
    assessAndEnhanceImage(rawImage, focusThresh, entropyThresh)
% assessAndEnhanceImage: BACKWARD-COMPATIBLE 6-output wrapper around the
% canonical assessFundusQuality.m implementation (single ROI / focus /
% entropy / enhancement core lives there since the P1+P2+P3 hardening).
%
% Numeric behavior is identical to the pre-inversion implementation for the
% same explicit thresholds: same validation, ROI, scores, enhancement.
% Extended FAIL reasons (degenerate coverage, severe exposure) and
% BORDERLINE semantics live in the canonical report - this wrapper
% returns the legacy threshold-only boolean so existing callers
% (runScreeningPipeline gate, preprocessFundusForSegmentation,
% train_DR_Grader fusion build, calibrator scorer) behave exactly as
% before. New code must call assessFundusQuality directly.
%
% INPUTS:
%   rawImage      - fundus photo, RGB or grayscale, uint8 or double.
%   focusThresh   - minimum focus score (default: qualityConfig).
%   entropyThresh - minimum illumination score (default: qualityConfig).
%                   Pass -Inf/-Inf for enhance-only (never rejects).
%
% OUTPUTS: isGradeable (legacy threshold-only boolean), enhancedRGB,
%   enhancedGray (background masked to 0), focusScore, entropyScore,
%   roiMask (reuse downstream instead of re-deriving).
%
% Requires: Image Processing Toolbox (via assessFundusQuality).

cfg0 = qualityConfig();
if nargin < 3 || isempty(entropyThresh)
    entropyThresh = cfg0.entropyThresh;
end
if nargin < 2 || isempty(focusThresh)
    focusThresh = cfg0.focusThresh;
end

% Legacy validation first so error identifiers stay exactly as before.
if isempty(rawImage) || ~isnumeric(rawImage) && ~islogical(rawImage)
    error('assessAndEnhanceImage:badInput', 'rawImage must be a non-empty numeric/logical image.');
end
if ndims(rawImage) ~= 2 && ndims(rawImage) ~= 3
    error('assessAndEnhanceImage:badInput', 'rawImage must be HxW or HxWxC.');
end
if size(rawImage,3) > 3
    warning('assessAndEnhanceImage:extraChannels', ...
        'Got %d channels; keeping first 3 (RGB) and ignoring alpha/depth extras.', size(rawImage,3));
    rawImage = rawImage(:,:,1:3);
end

% Calibration bypassed: explicit caller thresholds are authoritative here
% (the gate resolves calibration BEFORE calling - see runScreeningPipeline).
rep = assessFundusQuality(rawImage, 'FocusThresh', focusThresh, ...
    'EntropyThresh', entropyThresh, 'Config', cfg0, 'CalibrationDir', []);

isGradeable  = rep.legacyGradeable;
enhancedRGB  = rep.enhancedRGB;
enhancedGray = rep.enhancedGray;
focusScore   = rep.focusScore;
entropyScore = rep.entropyScore;
roiMask      = rep.roiMask;
end
