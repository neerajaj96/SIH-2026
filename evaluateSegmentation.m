function metrics = evaluateSegmentation(predMask, gtMask)
% evaluateSegmentation: Dice, IoU, sensitivity, specificity and precision
% for a single predicted-vs-ground-truth binary mask pair, or averaged
% over a whole set of pairs. This is Phase 3 of the earlier project
% review's fix-order plan ("Produce actual: Dice / IoU / Sensitivity /
% Specificity / Precision for each segmentation task") - nothing in the
% project computed these before; train_UNet_Segmentation.m's Tversky
% loss trains TOWARD a good Dice-like score but nothing ever reported an
% actual one on held-out data.
%
% VERIFIED: on a hand-checkable 10x10 synthetic case (a 4x4 ground-truth
% square and a 4x4 predicted square shifted by 1 pixel, giving a 3x3=9px
% overlap out of 16px in each mask), this reproduces
% Dice=0.5625, IoU=0.3913, Sens=0.5625, Spec=0.9167, Prec=0.5625 -
% independently computed in Python before this file was written, and
% matching the textbook formulas exactly. runSelfTests.m reruns this
% exact case in MATLAB.
%
% USAGE (single pair):
%   metrics = evaluateSegmentation(predMask, gtMask)
%
% USAGE (whole test set, batch form):
%   metrics = evaluateSegmentation(predMaskCellArray, gtMaskCellArray)
%   - pass two cell arrays of equal length, one mask per cell. Returns
%     per-image metrics AND the mean +/- std across images, which is what
%     you actually want to report (a single pooled Dice across all pixels
%     of all images lets your largest images dominate the average; the
%     per-image-then-average convention is standard in the segmentation
%     literature and is what this uses instead).
%
% INPUTS:
%   predMask, gtMask - logical masks, same size (or matched cell arrays
%                       of logical masks, one predicted/one ground-truth
%                       pair per index)
%
% OUTPUTS:
%   metrics - struct with fields .dice .iou .sensitivity .specificity
%             .precision (scalars, for a single pair); for the batch
%             form, instead a struct with .perImage (struct array),
%             .mean, .std (each with the same 5 fields) and .n
%
% Requires: nothing beyond base MATLAB.

if iscell(predMask)
    if ~iscell(gtMask) || numel(predMask) ~= numel(gtMask)
        error('evaluateSegmentation:cellMismatch', ...
            'predMask and gtMask must be cell arrays of the SAME length, one mask pair per index.');
    end
    n = numel(predMask);
    perImage = repmat(struct('dice',0,'iou',0,'sensitivity',0,'specificity',0,'precision',0), n, 1);
    for i = 1:n
        perImage(i) = localSinglePairMetrics(predMask{i}, gtMask{i});
    end
    fieldsToAgg = {'dice','iou','sensitivity','specificity','precision'};
    meanS = struct(); stdS = struct();
    for f = 1:numel(fieldsToAgg)
        vals = [perImage.(fieldsToAgg{f})];
        meanS.(fieldsToAgg{f}) = mean(vals, 'omitnan');
        stdS.(fieldsToAgg{f})  = std(vals, 'omitnan');
    end
    metrics = struct('perImage', {perImage}, 'mean', meanS, 'std', stdS, 'n', n);
    fprintf('=== Segmentation evaluation over %d image(s) ===\n', n);
    fprintf('%-12s %8s %8s\n', '', 'mean', 'std');
    for f = 1:numel(fieldsToAgg)
        fprintf('%-12s %8.4f %8.4f\n', fieldsToAgg{f}, meanS.(fieldsToAgg{f}), stdS.(fieldsToAgg{f}));
    end
else
    if ~isequal(size(predMask), size(gtMask))
        error('evaluateSegmentation:sizeMismatch', ...
            'predMask (%s) and gtMask (%s) must be the same size - resize/align them first (see runSegmentationNet.m for the resize-back convention this pipeline uses).', ...
            mat2str(size(predMask)), mat2str(size(gtMask)));
    end
    metrics = localSinglePairMetrics(predMask, gtMask);
end
end

% ------------------------------------------------------------------
function m = localSinglePairMetrics(pred, gt)
pred = logical(pred); gt = logical(gt);
TP = nnz(pred & gt);
FP = nnz(pred & ~gt);
FN = nnz(~pred & gt);
TN = nnz(~pred & ~gt);

m.dice = localSafeDiv(2*TP, 2*TP + FP + FN, 1.0); % by convention here, an empty-vs-empty pair (TP=FP=FN=0) counts as a PERFECT match, not undefined
m.iou  = localSafeDiv(TP, TP + FP + FN, 1.0);
m.sensitivity = localSafeDiv(TP, TP + FN, NaN); % undefined (NaN, not 0) if there's no positive ground truth to detect at all
m.specificity = localSafeDiv(TN, TN + FP, NaN);
m.precision   = localSafeDiv(TP, TP + FP, NaN); % undefined if the model predicted no positives at all
end

% ------------------------------------------------------------------
function v = localSafeDiv(num, den, valueIfZeroDenom)
if den > 0
    v = num / den;
else
    v = valueIfZeroDenom;
end
end
