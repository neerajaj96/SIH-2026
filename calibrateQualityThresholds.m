function [focusThresh, entropyThresh] = calibrateQualityThresholds(goodImageDir, badImageDir, varargin)
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
%   [...] = calibrateQualityThresholds(goodDir, badDir, 'OutputMat', 'qualityThresholds.mat')
%
% INPUTS:
%   goodImageDir - folder of fundus photos a human grader marked gradeable.
%   badImageDir  - (optional) folder of photos marked ungradeable (blurry,
%                  poorly lit). If omitted, the 5th percentile of the good
%                  set's own score distribution is used instead - weaker,
%                  but still better than an arbitrary constant.
%   'OutputMat'  - (optional) path for qualityThresholds.mat artifact.
%                  Default: qualityThresholds.mat in pwd. Pass '' to skip
%                  writing (print-only, backward compatible).
%
% OUTPUTS:
%   focusThresh, entropyThresh - pass these into assessAndEnhanceImage, or
%   better: let qualityLoadCalibration pick up the written .mat automatically.
%   Re-run this after any change to camera hardware or JPEG compression
%   settings - both shift these scores independent of actual image quality.
%
% Requires: Image Processing Toolbox, and assessAndEnhanceImage.m on the path.

if nargin < 2
    badImageDir = '';
end
p = inputParser;
addParameter(p, 'OutputMat', fullfile(pwd, qualityConfig().calibratedMatName), @(x) isempty(x) || ischar(x) || isstring(x));
parse(p, varargin{:});
outputMat = char(p.Results.OutputMat);

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
    % Simple separating cut with reported sens/spec/AUC so the operator
    % can judge overlap. For cost-sensitive tradeoffs, sweep the cut
    % along the ROC yourself - this reports the numbers to do so.
    focusThresh   = (min(goodFocus) + max(badFocus)) / 2;
    entropyThresh = (min(goodEntropy) + max(badEntropy)) / 2;
    fprintf('Calibrated from %d gradeable + %d ungradeable images.\n', numel(goodFiles), numel(badFiles));
    focusRep = localSeparationReport(goodFocus, badFocus, focusThresh, 'focus');
    entropyRep = localSeparationReport(goodEntropy, badEntropy, entropyThresh, 'entropy');
    calibrationReport = struct('mode', 'two-sided', 'nGood', numel(goodFiles), 'nBad', numel(badFiles), ...
        'focus', focusRep, 'entropy', entropyRep, 'date', datestr(now, 30), ...
        'configVersion', qualityConfig().version);
else
    focusThresh   = prctile(goodFocus, 5);
    entropyThresh = prctile(goodEntropy, 5);
    fprintf(['Calibrated from %d gradeable images only (5th-percentile rule) - ' ...
             'this only rejects the worst ~5%% of images you already trust are usable. ' ...
             'Add a folder of known-ungradeable images for a sharper, more meaningful cut.\n'], numel(goodFiles));
    calibrationReport = struct('mode', 'good-only-percentile', 'nGood', numel(goodFiles), 'nBad', 0, ...
        'date', datestr(now, 30), 'configVersion', qualityConfig().version);
end

fprintf('Suggested focusThresh   = %.2f  (observed range on gradeable set: %.2f - %.2f)\n', ...
    focusThresh, min(goodFocus), max(goodFocus));
fprintf('Suggested entropyThresh = %.2f  (observed range on gradeable set: %.2f - %.2f)\n', ...
    entropyThresh, min(goodEntropy), max(goodEntropy));

if ~isempty(outputMat)
    try
        save(outputMat, 'focusThresh', 'entropyThresh', 'calibrationReport');
        fprintf('Wrote calibration artifact: %s (load via qualityLoadCalibration).\n', outputMat);
        try
            metaPath = [outputMat '.meta.json'];
            fid = fopen(metaPath, 'w');
            if fid > 0
                fprintf(fid, '{"focusThresh":%.4f,"entropyThresh":%.4f,"mode":"%s","nGood":%d,"nBad":%d,"date":"%s","configVersion":"%s"}', ...
                    focusThresh, entropyThresh, calibrationReport.mode, calibrationReport.nGood, calibrationReport.nBad, calibrationReport.date, calibrationReport.configVersion);
                fclose(fid);
                fprintf('Wrote sidecar: %s\n', metaPath);
            end
        catch
        end
    catch ME
        warning('calibrateQualityThresholds:saveFailed', 'Could not write %s (%s).', outputMat, ME.message);
    end
end
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

function rep = localSeparationReport(goodV, badV, thresh, name)
% Sensitivity = P(good >= thresh); specificity = P(bad < thresh).
% AUC via Mann-Whitney rank statistic (no toolbox needed).
sens = mean(goodV >= thresh);
spec = mean(badV < thresh);
overlap = max(badV) > min(goodV);
nG = numel(goodV); nB = numel(badV);
allV = [goodV(:); badV(:)];
[~, order] = sort(allV);
ranks = zeros(size(allV));
ranks(order) = 1:numel(allV);
rankGood = sum(ranks(1:nG));
auc = (rankGood - nG*(nG+1)/2) / (nG*nB);
if overlap
    warning('calibrateQualityThresholds:overlap', ...
        '%s distributions overlap (max bad %.2f > min good %.2f): midpoint cut misclassifies; collect harder negatives or move operating point.', ...
        name, max(badV), min(goodV));
end
fprintf('  %s: sens=%.3f spec=%.3f AUC=%.3f gap=[minGood %.2f vs maxBad %.2f]%s\n', ...
    name, sens, spec, auc, min(goodV), max(badV), ternary(overlap, ' OVERLAP', ''));
rep = struct('thresh', thresh, 'sens', sens, 'spec', spec, 'auc', auc, ...
    'minGood', min(goodV), 'maxGood', max(goodV), 'minBad', min(badV), 'maxBad', max(badV), 'overlap', overlap);
end

function s = ternary(cond, a, b)
if cond, s = a; else, s = b; end
end
