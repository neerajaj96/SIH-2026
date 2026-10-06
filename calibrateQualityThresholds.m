function [focusThresh, entropyThresh] = calibrateQualityThresholds(goodImageDir, badImageDir)
% calibrateQualityThresholds: Derives IQA gate thresholds from labeled
% example images instead of guessing a constant.
%
% The original pipeline called assessAndEnhanceImage(img, 40, 3.5) and
% described 40/3.5 as "statistically derived... optimized by analyzing
% the distribution of the APTOS 2019 dataset" - but nothing in the
% codebase ever derived them from anything; they were just literals at
% the call site. On one real, visibly-sharp, JPEG-compressed sample
% fundus photo, the focus score came out ~10 - about 4x under that "40"
% - so treat that number as a placeholder, not a setting. This script is
% the actual derivation step that was missing.
%
% USAGE:
%   [focusThresh, entropyThresh] = calibrateQualityThresholds('data/gradeable')
%   [focusThresh, entropyThresh] = calibrateQualityThresholds('data/gradeable', 'data/ungradeable')
%
% INPUTS:
%   goodImageDir - folder of fundus photos a human grader marked gradeable.
%   badImageDir  - (optional) folder of photos marked ungradeable (blurry,
%                  poorly lit). If omitted, the 5th percentile of the good
%                  set's own score distribution is used instead - weaker,
%                  but still better than an arbitrary constant.
%
% OUTPUTS:
%   focusThresh, entropyThresh - pass these into assessAndEnhanceImage.
%   Re-run this after any change to camera hardware or JPEG compression
%   settings - both shift these scores independent of actual image quality.
%
% Requires: Image Processing Toolbox, and assessAndEnhanceImage.m on the path.

if nargin < 2
    badImageDir = '';
end

goodFiles = localListImages(goodImageDir);
if isempty(goodFiles)
    error('calibrateQualityThresholds:noImages', 'No images found in %s', goodImageDir);
end

[goodFocus, goodEntropy] = localScoreAll(goodFiles);

if ~isempty(badImageDir) && isfolder(badImageDir)
    badFiles = localListImages(badImageDir);
    if isempty(badFiles)
        warning('No images found in %s - falling back to percentile-only calibration.', badImageDir);
        badImageDir = '';
    else
        [badFocus, badEntropy] = localScoreAll(badFiles);
    end
end

if ~isempty(badImageDir) && isfolder(badImageDir)
    % Midpoint between the worst "good" score and the best "bad" score.
    % This is a simple separating cut, not an ROC-optimal one - if you
    % care about trading sensitivity against false rejects differently,
    % compute an ROC curve from goodFocus/badFocus yourself and pick a
    % different operating point.
    focusThresh   = (min(goodFocus) + max(badFocus)) / 2;
    entropyThresh = (min(goodEntropy) + max(badEntropy)) / 2;
    fprintf('Calibrated from %d gradeable + %d ungradeable images.\n', numel(goodFiles), numel(badFiles));
else
    focusThresh   = prctile(goodFocus, 5);
    entropyThresh = prctile(goodEntropy, 5);
    fprintf(['Calibrated from %d gradeable images only (5th-percentile rule) - ' ...
             'this only rejects the worst ~5%% of images you already trust are usable. ' ...
             'Add a folder of known-ungradeable images for a sharper, more meaningful cut.\n'], numel(goodFiles));
end

fprintf('Suggested focusThresh   = %.2f  (observed range on gradeable set: %.2f - %.2f)\n', ...
    focusThresh, min(goodFocus), max(goodFocus));
fprintf('Suggested entropyThresh = %.2f  (observed range on gradeable set: %.2f - %.2f)\n', ...
    entropyThresh, min(goodEntropy), max(goodEntropy));
end

% ------------------------------------------------------------------
function files = localListImages(folder)
files = [dir(fullfile(folder,'*.jpg')); dir(fullfile(folder,'*.jpeg')); ...
         dir(fullfile(folder,'*.png')); dir(fullfile(folder,'*.tif'))];
end

function [focusScores, entropyScores] = localScoreAll(files)
focusScores = zeros(numel(files),1);
entropyScores = zeros(numel(files),1);
for i = 1:numel(files)
    img = imread(fullfile(files(i).folder, files(i).name));
    [~, ~, ~, focusScores(i), entropyScores(i), ~] = assessAndEnhanceImage(img, -Inf, -Inf);
end
end
