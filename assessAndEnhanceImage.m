function [isGradeable, enhancedRGB, enhancedGray, focusScore, entropyScore, roiMask] = ...
    assessAndEnhanceImage(rawImage, focusThresh, entropyThresh)
% assessAndEnhanceImage: Evaluates fundus image quality within the actual
% fundus circle (not the black camera surround) and returns both a
% single-channel and an RGB contrast-enhanced version.
%
% FIXES vs. the original draft:
%   - rgb2hsv(rawImage) used to be called unconditionally even though the
%     function had separate dead branches for grayscale input elsewhere -
%     feeding it a true grayscale image would have crashed on that line.
%     Grayscale is now normalized to pseudo-RGB up front, once.
%   - Focus/entropy were computed over the WHOLE frame, including the
%     black background outside the circular fundus image and any corner
%     artifacts (light leaks, sticker tags). Both are now computed only
%     within a detected fundus-circle ROI.
%   - Background illumination correction used imsubtract (additive, clips
%     to 0 in dim regions). Now uses a divide-based flat-field correction,
%     which doesn't crush the already-dim periphery to black.
%   - Only a single-channel, green-derived enhanced image was returned,
%     and downstream code (production_inference.m) was fusing the RAW
%     image instead of it anyway. This now returns a genuine enhanced RGB
%     image too, so the classifier can actually be fed enhanced input.
%
% ADDED THIS ROUND:
%   - The flat-field background structuring element used to be a FIXED
%     strel('disk', 65) - a constant in absolute pixels. That's fine for
%     one camera at one resolution, but this project's own datasets span
%     a huge resolution range (DRIVE ~584x565, a typical portable-camera
%     capture ~1280x1280, IDRiD 4288x2848) - a fixed 65px disk is a
%     reasonable fraction of a small fundus circle and a tiny,
%     near-useless fraction of a large one, so the "background" estimate
%     it produced was not comparable across datasets/cameras. It's now
%     sized as a fraction of the DETECTED ROI diameter instead. The 5%
%     fraction was chosen so that on the one real sample photo this
%     project has actually validated against (~1280x1280, ROI diameter
%     ~1150px per the focus-score calibration note below), this
%     reproduces close to the original 65px constant (0.05 x 1150 = 58px)
%     - i.e. this generalizes the one behavior that's actually been
%     eyeballed on a real photo, rather than replacing it with an
%     unrelated new guess. It has NOT been validated across the other
%     three datasets - if illumination correction looks visibly different
%     on IDRiD/DRIVE/Messidor-2 samples, that fraction is the first thing
%     to re-tune, not the CLAHE/denoise parameters.
%     This directly matters for the plan to train on APTOS/IDRiD/DRIVE
%     and test on Messidor-2 (different cameras, different resolutions) -
%     without this fix, "enhancement" would have behaved inconsistently
%     across the very datasets this project's own validation strategy
%     depends on comparing.
%
% INPUTS:
%   rawImage      - fundus photo, RGB or grayscale, uint8 or double.
%   focusThresh   - minimum acceptable focus score. THIS IS A PLACEHOLDER,
%                   NOT A DERIVED VALUE - see calibrateQualityThresholds.m.
%                   On a real, visibly-sharp, JPEG-compressed 1280x1280
%                   sample photo, this function's own focus score came out
%                   around 10-11; the default below is set closer to that
%                   observed range than the original "40" was, but it is
%                   still a guess and needs real calibration before you
%                   trust the accept/reject gate at deployment.
%   entropyThresh - minimum acceptable illumination-quality score.
%
% OUTPUTS:
%   isGradeable   - true/false.
%   enhancedRGB   - contrast-enhanced RGB image, background masked to 0.
%                   Fuse THIS with the lesion/vessel masks downstream, not
%                   rawImage.
%   enhancedGray  - contrast-enhanced single-channel (green-derived) image,
%                   background masked to 0. Convenient single-channel input
%                   for the segmentation networks.
%   focusScore, entropyScore - raw scores, returned even on a fail so the
%                   caller can see how far off threshold a rejected image was.
%   roiMask       - logical mask of the detected fundus circle. Reuse this
%                   downstream instead of re-deriving it.
%
% Requires: Image Processing Toolbox

cfg0 = qualityConfig();
if nargin < 3 || isempty(entropyThresh)
    entropyThresh = cfg0.entropyThresh;
end
if nargin < 2 || isempty(focusThresh)
    focusThresh = cfg0.focusThresh; % see the calibration note above and in the header comment
end

% --- 0. Normalize to RGB up front so every line after this can assume
%        3 channels; removes the old dead grayscale-handling branches
%        that rgb2hsv would have crashed straight through anyway. ---
if size(rawImage,3) == 1
    rawImage = repmat(rawImage, [1 1 3]);
end
rawImage = im2uint8(rawImage);

% --- 1. Fundus-circle ROI ---
% A naive whole-frame quality score is biased by however much black
% border a given camera produces, and by any bright non-retinal artifact
% near the edges. Restrict both the quality metrics and the enhancement
% to the actual fundus circle.
grayFull = rgb2gray(rawImage);
roiMask = grayFull > cfg0.roiSeedThresh;
roiMask = imfill(roiMask, 'holes');
cc = bwconncomp(roiMask);
if cc.NumObjects > 1
    % Largest component wins: dust specks, sticker tags, and light-leak
    % corners are orders of magnitude smaller than the fundus disc, so
    % picking the max implicitly rejects anything below speckFrac scale.
    sizes = cellfun(@numel, cc.PixelIdxList);
    [~, biggest] = max(sizes);
    roiMask = false(size(roiMask));
    roiMask(cc.PixelIdxList{biggest}) = true;
end
% Edge-shave scales with resolution: floor 8px preserves validated 1280px
% behaviour (~8px), large IDRiD frames get proportionally more.
roiDiaEst = sqrt(4 * nnz(roiMask) / pi);
erodeR = max(cfg0.roiErodeDisk, round(cfg0.roiErodeRelFrac * roiDiaEst));
roiMask = imerode(roiMask, strel('disk', erodeR));
if ~any(roiMask(:))
    roiMask = true(size(grayFull)); % degenerate input (all-black/all-white) - fall back rather than crash
end

% --- 2. Quality assessment, restricted to the ROI ---
laplacianFilter = fspecial('laplacian', cfg0.laplacianAlpha);
laplacianImage = imfilter(double(grayFull), laplacianFilter, 'replicate');
focusScore = var(laplacianImage(roiMask));

hsvImage = rgb2hsv(rawImage);
vChannel = hsvImage(:,:,3);
entropyScore = localMaskedEntropy(vChannel, roiMask, cfg0.entropyBins);

isGradeable = (focusScore >= focusThresh) && (entropyScore >= entropyThresh);

% --- 3. Adaptive enhancement ---
% Green channel carries the strongest vessel/lesion contrast (hemoglobin
% absorbs green light), so illumination correction is derived from it...
%
% Structuring element sized as a fraction of the detected ROI diameter,
% not a fixed pixel count - see the "ADDED THIS ROUND" header note for
% why and what it reproduces.
roiDiameter = sqrt(4 * nnz(roiMask) / pi); % equivalent-circle diameter of the detected fundus ROI
bgRadius = max(round(cfg0.bgFraction * roiDiameter), cfg0.bgFloorPx); % fraction of ROI diameter, floored so tiny/degenerate ROIs don't collapse
structuringElement = strel('disk', bgRadius);

greenChannel = rawImage(:,:,2);
greenBackground = imopen(greenChannel, structuringElement);
% ...via DIVISION (flat-fielding), not subtraction. Subtraction clips
% straight to 0 in already-dim areas (common in the fundus periphery);
% division rescales instead of crushing them to black.
enhancedGray = localFlatFieldClaheDenoise(greenChannel, greenBackground, roiMask, cfg0);
enhancedGray(~roiMask) = 0;

% ...then the same correction is applied per-channel so the classifier
% gets a genuinely enhanced RGB image (the original pipeline silently
% fused the RAW image at this point instead - see production_inference.m).
enhancedRGB = rawImage;
for c = 1:3
    chan = rawImage(:,:,c);
    bg = imopen(chan, structuringElement);
    enhancedRGB(:,:,c) = localFlatFieldClaheDenoise(chan, bg, roiMask, cfg0);
end
for c = 1:3
    chanMasked = enhancedRGB(:,:,c);
    chanMasked(~roiMask) = 0;
    enhancedRGB(:,:,c) = chanMasked;
end

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
out = imnlmfilt(claheChan);
end

% ------------------------------------------------------------------
function e = localMaskedEntropy(channel01, mask, nBins)
% Shannon entropy (base 2, nBins-bin histogram), restricted to mask==true.
% The built-in entropy() function has no mask input, so this reimplements
% its histogram/entropy math narrowly over just the ROI pixels.
if nargin < 3 || isempty(nBins), nBins = 256; end
vals = channel01(mask);
counts = histcounts(vals, nBins, 'BinLimits', [0 1]);
p = counts / sum(counts);
p = p(p > 0);
e = -sum(p .* log2(p));
end
