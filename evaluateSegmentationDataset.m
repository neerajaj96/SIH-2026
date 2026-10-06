function results = evaluateSegmentationDataset(netOrMat, imagePaths, maskPaths, varargin)
% evaluateSegmentationDataset: End-to-end held-out evaluator that closes
% the "Tversky trains toward Dice but nothing ever reports one" gap.
%
% Runs real inference (assessAndEnhanceImage -> runSegmentationNet) on
% paired raw images + ground-truth masks, then scores with
% evaluateSegmentation.m (macro mean/std + pooled micro + empty-GT audit).
% Ground truth is used ONLY for scoring AFTER prediction - never fed into
% the network (no inference-time leakage by construction).
%
% INPUTS:
%   netOrMat   - trained dlnetwork OR path to unet_*.mat (loads 'net')
%   imagePaths - cell array of raw fundus image files
%   maskPaths  - cell array of GT binary mask files (0/255 or 0/1)
%   Name/Value:
%     'TargetName' - for messages (default 'segmentation')
%     'NetInputSize' - default segmentationConfig.inputSize
%     'SaveDir'    - optional folder for per-image CSV + failure overlays
%                    (default '' = no files written)
%     'WorstK'     - # worst-Dice cases to visualize if SaveDir given
%                    (default 8)
%
% OUTPUT:
%   results - struct with .metrics (evaluateSegmentation batch output),
%             .perImagePaths, .predSizes, .gtSizes, .resizedGT (logical per
%             pair whether GT was resized to match prediction)
%
% Requires: Image Processing Toolbox, Deep Learning Toolbox.

p = inputParser();
p.addParameter('TargetName', 'segmentation', @(s) ischar(s) || isstring(s));
p.addParameter('NetInputSize', [], @isnumeric);
p.addParameter('SaveDir', '', @(s) ischar(s) || isstring(s));
p.addParameter('WorstK', 8, @isnumeric);
p.parse(varargin{:});
opt = p.Results;
targetName = char(opt.TargetName);
cfg = segmentationConfig();
if isempty(opt.NetInputSize), opt.NetInputSize = cfg.inputSize; end

if ischar(netOrMat) || isstring(netOrMat)
    S = load(char(netOrMat), 'net');
    net = S.net;
else
    net = netOrMat;
end

n = numel(imagePaths);
assert(numel(maskPaths) == n, ...
    'evaluateSegmentationDataset:pairMismatch', ...
    'imagePaths (%d) vs maskPaths (%d) length mismatch.', n, numel(maskPaths));
assert(n >= 1, 'evaluateSegmentationDataset:empty', 'No pairs to evaluate.');

predCells = cell(n, 1); gtCells = cell(n, 1);
resizedGT = false(n, 1);
for i = 1:n
    raw = imread(imagePaths{i});
    % Never rejects here (-Inf): evaluation must score every held-out
    % image, not silently drop hard cases via the quality gate. The gate's
    % accept/reject behavior is a separate Stage-1 concern.
    [~, ~, enhancedGray, ~, ~, roiMask] = assessAndEnhanceImage(raw, -Inf, -Inf);
    [maskOrig, ~] = runSegmentationNet(net, enhancedGray, roiMask, opt.NetInputSize);

    g = imread(maskPaths{i});
    if ndims(g) == 3, g = g(:,:,1); end
    g = double(g);
    if max(g(:)) == min(g(:))
        gb = logical(g ~= 0);
    else
        gb = g > (max(g(:)) + min(g(:))) / 2; % handles 0/1 and 0/255
    end
    if ~isequal(size(gb), size(maskOrig))
        warning(['evaluateSegmentationDataset:sizeMismatch - GT %s is %s but prediction is %s. ' ...
                 'Resizing GT with nearest to match (categorical). Check pairing if this happens often.'], ...
            maskPaths{i}, mat2str(size(gb)), mat2str(size(maskOrig)));
        gb = imresize(gb, size(maskOrig), 'nearest');
        resizedGT(i) = true;
    end
    predCells{i} = logical(maskOrig);
    gtCells{i} = logical(gb);
    if mod(i, 25) == 0 || i == n
        fprintf('  %s eval: %d/%d inferred\n', targetName, i, n);
    end
end

metrics = evaluateSegmentation(predCells, gtCells);
results = struct('metrics', metrics, 'perImagePaths', {imagePaths}, ...
    'resizedGT', resizedGT, 'targetName', targetName, ...
    'predCells', {predCells}, 'gtCells', {gtCells});
fprintf('%s held-out: n=%d mean Dice=%.4f IoU=%.4f Sens=%.4f Spec=%.4f Prec=%.4f (pooled Dice=%.4f). %d empty GT (%d correct).\n', ...
    targetName, metrics.n, metrics.mean.dice, metrics.mean.iou, ...
    metrics.mean.sensitivity, metrics.mean.specificity, metrics.mean.precision, ...
    metrics.pooled.dice, metrics.nEmpty, metrics.nEmptyCorrect);

if ~isempty(opt.SaveDir)
    if ~isfolder(opt.SaveDir), mkdir(opt.SaveDir); end
    % Per-image CSV for cross-dataset diagnostics.
    diceVals = arrayfun(@(m) m.dice, metrics.perImage);
    T = table(imagePaths(:), maskPaths(:), diceVals(:), ...
        arrayfun(@(m) m.iou, metrics.perImage), ...
        arrayfun(@(m) m.sensitivity, metrics.perImage), ...
        arrayfun(@(m) m.specificity, metrics.perImage), ...
        arrayfun(@(m) m.precision, metrics.perImage), resizedGT, ...
        'VariableNames', {'image','gtMask','dice','iou','sensitivity','specificity','precision','gtResized'});
    csvPath = fullfile(opt.SaveDir, sprintf('%s_eval_%s.csv', targetName, datestr(now,'yyyymmdd_HHMMSS')));
    writetable(T, csvPath);
    fprintf('Wrote %s\n', csvPath);
    visualizeSegmentationFailures(predCells, gtCells, imagePaths, opt.SaveDir, opt.WorstK);
end
end
