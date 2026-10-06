function cfg = segmentationConfig()
% segmentationConfig: SINGLE source of truth for every segmentation
% constant that was previously duplicated across train_UNet_Segmentation.m,
% preprocessFundusForSegmentation.m and runSegmentationNet.m.
%
% Stage-2 fix for the "one logical constant in three places" defect:
% changing input resolution previously required editing three files in
% sync with no check. Now all three call this. If you change inputSize,
% everything follows - training datastore, U-Net definition, inference
% resize-down and resize-back.
%
% INPUT resolution policy (per Stage-2 review decision):
%   - Locked at 512x512 for all training/inference now.
%   - Designed for later benchmarking (512 vs 768 for small-MA recall):
%     pass cfg = segmentationConfig(); cfg.inputSize = [768 768]; to
%     experiment - every consumer reads cfg, so no other edit is needed.
%     Report resolution in every .meta.json (see saveModelWithMetadata).
%
% MASK CONVENTION (critical fix documented here, enforced in
% buildSegmentationFileLists.m + validateMaskConventions.m):
%   - Canonical on-disk: binary PNG, 0 = background, 255 = foreground.
%   - pixelLabelDatastore labelIDs therefore MUST be [0 255], NOT [0 1].
%     The previous [0 1] silently mismatched every real 0/255 mask.
%   - Masks stored as 0/1 are auto-detected and converted (x255) at
%     file-list build time with a logged warning, not silently trained on.
%
% OUTPUT: cfg struct with:
%   .inputSize      [H W] network input (default [512 512])
%   .imageSize      [H W C] U-Net imageSize (default [512 512 1])
%   .numClasses     2 (Background, Foreground)
%   .classNames     ["Background","Foreground"] - order MUST match
%                   runSegmentationNet.m's classIdx==2 => Foreground rule
%   .labelIDs       [0 255] - on-disk pixel values (see above)
%   .encoderDepth   4
%   .valFraction    0.15
%   .seed           42 - RNG seed for splits + init (logged in metadata)
%   .tverskyAlpha/Beta per target (vessels, MA/HE, exudates)
%   .minBlobAreaPx  6 - placeholder-pathology filter floor (matches
%                   runScreeningPipeline.m simulated fallback)
%   .fusionSize     [224 224] - grader fusion tensor (Stage-3 contract;
%                   recorded here so seg callers and grader callers agree)
%
% Requires: nothing beyond base MATLAB.

cfg = struct();
cfg.inputSize   = [512 512];
cfg.imageSize   = [512 512 1];
cfg.numClasses  = 2;
cfg.classNames  = ["Background","Foreground"];
cfg.labelIDs    = [0 255];
cfg.encoderDepth = 4;
cfg.valFraction = 0.15;
cfg.seed        = 42;

% Tversky alpha (FP weight) / beta (FN weight). beta > alpha recalls
% rare foreground (vessels ~2% of pixels, MA far rarer) at the cost of
% extra false positives - appropriate for screening.
cfg.tversky = struct( ...
    'vessels', struct('alpha', 0.3, 'beta', 0.7), ...
    'mahe',    struct('alpha', 0.3, 'beta', 0.7), ...
    'exudate', struct('alpha', 0.3, 'beta', 0.6));

cfg.minBlobAreaPx = 6;
cfg.fusionSize    = [224 224];

% Training hyperparams (moved here from inline literals so metadata +
% resume logic can record/verify them instead of trusting comments).
cfg.train = struct( ...
    'initialLearnRate', 1e-3, ...
    'maxEpochs', 30, ...
    'miniBatchSize', 8, ...
    'learnRateDropFactor', [], ... % adam default schedule; empty = none
    'checkpointFrequency', 5, ...
    'checkpointFrequencyUnit', 'epoch', ...
    'validationFrequency', 20);
end
