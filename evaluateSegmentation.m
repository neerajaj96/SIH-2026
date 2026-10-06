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
%     per-image metrics AND the mean +/- std across images (macro average,
%     the literature standard - every image counts equally), PLUS a pooled
%     (micro) average over all pixels (every pixel counts equally).
%     Report BOTH: if macro >> micro, small images with empty masks are
%     inflating the mean; if micro >> macro, large images dominate.
%     See evaluateSegmentationDataset.m for the end-to-end held-out runner.
%
% EMPTY-MASK POLICY (Stage-2 documented, unchanged behavior):
%   - Empty-vs-empty (no GT lesion, no predicted lesion): dice=1, iou=1
%     (perfect - correctly predicted absence), sens/spec/prec=NaN
%     (undefined - no positive to detect). This rewards true negatives
%     without fabricating a sensitivity.
%   - Empty GT with FP>0: dice=0 (not NaN) - a false alarm on a healthy
%     image must hurt, not be excluded. NaN-mean aggregation would hide
%     this if you only looked at sensitivity.
%   - Always check .nEmpty and .nEmptyCorrect alongside the means: a
%     0.95 mean Dice with 80% empty images is a different claim than the
%     same mean with 5% empty images.
%
% INPUTS:
%   predMask, gtMask - logical masks, same size (or matched cell arrays
%                       of logical masks, one predicted/one ground-truth
%                       pair per index)
%
% OUTPUTS:
%   metrics - single-pair struct with .dice .iou .sensitivity .specificity
%             .precision plus audit counts .tp .fp .fn .tn .nEmptyGT
%             (1 if GT empty else 0); batch form struct with .perImage
%             (struct array, same fields), .mean, .std, .pooled (micro),
%             .n, .nEmpty, .nEmptyCorrect
%
% Requires: nothing beyond base MATLAB.

if iscell(predMask)
    if ~iscell(gtMask) || numel(predMask) ~= numel(gtMask)
        error('evaluateSegmentation:cellMismatch', ...
            'predMask and gtMask must be cell arrays of the SAME length, one mask pair per index.');
    end
    n = numel(predMask);
    perImage = repmat(struct('dice',0,'iou',0,'sensitivity',0,'specificity',0,'precision',0, ...
        'tp',0,'fp',0,'fn',0,'tn',0,'nEmptyGT',0), n, 1);
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
    % Pooled (micro): sum counts first, then compute - large images dominate.
    TP = sum([perImage.tp]); FP = sum([perImage.fp]);
    FN = sum([perImage.fn]); TN = sum([perImage.tn]);
    pooled = struct( ...
        'dice', localSafeDiv(2*TP, 2*TP + FP + FN, 1.0), ...
        'iou',  localSafeDiv(TP, TP + FP + FN, 1.0), ...
        'sensitivity', localSafeDiv(TP, TP + FN, NaN), ...
        'specificity', localSafeDiv(TN, TN + FP, NaN), ...
        'precision',   localSafeDiv(TP, TP + FP, NaN));
    nEmpty = sum([perImage.nEmptyGT]);
    % Empty-correct: empty GT AND dice==1 (no FP). Distinguishes "model
    % correctly says healthy" from "model alarms on every healthy image".
    nEmptyCorrect = sum(arrayfun(@(m) m.nEmptyGT == 1 && m.dice == 1.0, perImage));
    metrics = struct('perImage', {perImage}, 'mean', meanS, 'std', stdS, ...
        'pooled', pooled, 'n', n, 'nEmpty', nEmpty, 'nEmptyCorrect', nEmptyCorrect);
    fprintf('=== Segmentation evaluation over %d image(s) (%d empty GT, %d correctly empty) ===\n', n, nEmpty, nEmptyCorrect);
    fprintf('%-12s %8s %8s %8s\n', '', 'mean', 'std', 'pooled');
    for f = 1:numel(fieldsToAgg)
        fprintf('%-12s %8.4f %8.4f %8.4f\n', fieldsToAgg{f}, meanS.(fieldsToAgg{f}), stdS.(fieldsToAgg{f}), pooled.(fieldsToAgg{f}));
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

m.dice = localSafeDiv(2*TP, 2*TP + FP + FN, 1.0); % empty-vs-empty = 1.0 (correct absence); empty-GT-with-FP = 0 (false alarm hurts)
m.iou  = localSafeDiv(TP, TP + FP + FN, 1.0);
m.sensitivity = localSafeDiv(TP, TP + FN, NaN); % NaN if no positive GT - undefined, not 0
m.specificity = localSafeDiv(TN, TN + FP, NaN);
m.precision   = localSafeDiv(TP, TP + FP, NaN); % NaN if no predicted positive
m.tp = TP; m.fp = FP; m.fn = FN; m.tn = TN;
m.nEmptyGT = double(~any(gt(:)));
end

% ------------------------------------------------------------------
function v = localSafeDiv(num, den, valueIfZeroDenom)
if den > 0
    v = num / den;
else
    v = valueIfZeroDenom;
end
end
