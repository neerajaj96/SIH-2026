function visualizeSegmentationFailures(predCells, gtCells, imagePaths, saveDir, worstK)
% visualizeSegmentationFailures: Failure-case overlays for hostile review.
%
% For the worst-K Dice pairs (lowest Dice first), writes a side-by-side
% PNG: [raw fundus | TP green + FP red + FN blue overlay on enhanced gray].
% Green = correctly found, red = false alarm, blue = missed lesion.
% A model that only fails on 1px borders looks very different from one
% that misses whole lesions - this makes that visible without trusting a
% single mean Dice.
%
% INPUTS:
%   predCells, gtCells - matched cell arrays of logical masks (same size
%                        per pair; typically from evaluateSegmentationDataset)
%   imagePaths - raw image files (for the left panel; if unreadable, the
%                overlay panel is still written)
%   saveDir    - output folder (created if missing)
%   worstK     - # worst cases to render (default 8)
%
% Requires: Image Processing Toolbox (imread/imwrite only).

if nargin < 5 || isempty(worstK), worstK = 8; end
if ~isfolder(saveDir), mkdir(saveDir); end
n = numel(predCells);
assert(numel(gtCells) == n, 'visualizeSegmentationFailures:pairMismatch', 'pred/gt length mismatch.');

dices = zeros(n, 1);
for i = 1:n
    m = evaluateSegmentation(predCells{i}, gtCells{i});
    dices(i) = m.dice;
end
[~, order] = sort(dices, 'ascend');
k = min(worstK, n);
fprintf('Rendering %d worst-Dice overlays to %s (worst Dice=%.4f)...\n', k, saveDir, dices(order(1)));

for j = 1:k
    i = order(j);
    pred = logical(predCells{i}); gt = logical(gtCells{i});
    TP = pred & gt; FP = pred & ~gt; FN = ~pred & gt;
    % Enhanced-gray background for context (same path as inference).
    try
        raw = imread(imagePaths{i});
        [~, ~, enh, ~, ~, ~] = assessAndEnhanceImage(raw, -Inf, -Inf);
        bg = repmat(enh, [1 1 3]);
    catch
        bg = uint8(zeros([size(pred) 3]));
    end
    if ~isequal(size(bg,1), size(pred,1)) || ~isequal(size(bg,2), size(pred,2))
        bg = uint8(repmat(uint8(pred)*0 + 128, [1 1 3]));
    end
    overlay = bg;
    overlay(repmat(TP, [1 1 3])) = uint8(0);
    tmp = overlay(:,:,2); tmp(TP) = 255; overlay(:,:,2) = tmp; % green
    tmp = overlay(:,:,1); tmp(FP) = 255; overlay(:,:,1) = tmp; % red
    tmp = overlay(:,:,1); tmp(FP) = tmp(FP)*0 + 255; overlay(:,:,1) = tmp;
    tmp = overlay(:,:,2); tmp(FP) = 0; overlay(:,:,2) = tmp;
    tmp = overlay(:,:,3); tmp(FP) = 0; overlay(:,:,3) = tmp;
    tmp = overlay(:,:,3); tmp(FN) = 255; overlay(:,:,3) = tmp; % blue
    tmp = overlay(:,:,1); tmp(FN & ~FP) = 0; overlay(:,:,1) = tmp;
    tmp = overlay(:,:,2); tmp(FN & ~FP) = 0; overlay(:,:,2) = tmp;
    [~, stem, ~] = fileparts(imagePaths{i});
    outPath = fullfile(saveDir, sprintf('fail_%02d_dice%.3f_%s.png', j, dices(i), stem));
    imwrite(overlay, outPath);
end
fprintf('Wrote %d overlays (green=TP red=FP blue=FN).\n', k);
end
