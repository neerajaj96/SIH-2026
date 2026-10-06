function cfg = gradingConfig()
% gradingConfig: SINGLE source of truth for the DR severity grading
% contract (Stage-3). Freezes the five-channel fusion tensor that Stage-2
% fixed and Stage-3 consumes - so training, inference and tests cannot
% silently drift apart.
%
% FIVE-CHANNEL ORDER (frozen - do not reorder without updating every
% consumer + test):
%   ch1 = enhanced R (0-255, single, bilinear to [224 224])
%   ch2 = enhanced G (same)
%   ch3 = enhanced B (same)
%   ch4 = predicted vessel mask *255 (0/255, single, NEAREST to [224 224])
%   ch5 = predicted lesion mask *255, where lesion = MA/HE OR exudate
%         (0/255, single, NEAREST). Stage-2 contract lesionMask=mahe|exudate
%         (see SEGMENTATION_HANDOFF.md section 4); the mission's
%         "RGB + vessel + MA/HE + exudate" maps to this merged lesion
%         channel - NOT to 6 channels. A 6-channel RGB+vessel+mahe+exudate
%         redesign would break the pretrained first-conv expansion,
%         runScreeningPipeline.m inference and every saved .mat - rejected.
%
% DTYPE/RANGE/LAYOUT (frozen):
%   single HxWx5, channels 1-3 in [0,255] (enhancedRGB, NOT raw image -
%   raw bypass is a training bug, except the deliberate baseline ablation),
%   channels 4-5 in {0,255} (predicted masks, never ground-truth masks at
%   inference or fusion-training time). dlarray layout 'SSC'.
%   imageInputLayer([224 224 5], Normalization='zscore'), name 'fusion_input'.
%
% RESIZE/INTERP (frozen, Stage-2 fix):
%   photos bilinear, masks nearest-neighbor (categorical). Previous default
%   bicubic on masks invented fractional values the grader never saw binary.
%
% LABELS (frozen ICDR):
%   grades 0..4 = [No DR, Mild, Moderate, Severe, Proliferative].
%   classValues single(0:4)'. Classifier head 'dr_fc' exactly 5 outputs +
%   'prob' softmax. Predicted grade = argmax - 1. Referable = grade >= 2.
%
% LOSS DEFAULTS (audited, see ordinalGradingLoss.m):
%   hybrid cross-entropy + lambda * mean((E_pred - E_true)^2), lambda 0.5
%   default (tune ONLY on validation, never test).
%
% SPLITS/SEED (recorded, owned by buildGradingDatasets.m):
%   seed 42 everywhere; DDR case 85/15 train/val + Messidor-2 held-out test;
%   Messidor-only case 70/15/15 patient-level via pairs.csv.
%
% Requires: nothing beyond base MATLAB.

cfg = struct();
cfg.channelOrder = {'R','G','B','vessel','lesion'};
cfg.channelCount = 5;
cfg.lesionPolicy = 'mahe OR exudate (Stage-2 lesionMask)';
cfg.inputSize = [224 224];
cfg.inputChannels = 5;
cfg.inputLayerName = 'fusion_input';
cfg.photoInterp = 'bilinear';
cfg.maskInterp = 'nearest';
cfg.photoRange = [0 255];
cfg.maskValues = [0 255];
cfg.dtype = 'single';
cfg.dlarrayLayout = 'SSC';

cfg.numClasses = 5;
cfg.classValues = single(0:4)';
cfg.classNames = ["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","Proliferative DR"];
cfg.headFcName = 'dr_fc';
cfg.headProbName = 'prob';
cfg.referableThreshold = 2;

cfg.ordinalLambda = 0.5;
cfg.seed = 42;

cfg.train = struct( ...
    'initialLearnRate', 1e-4, ...
    'learnRateSchedule', 'piecewise', ...
    'learnRateDropFactor', 0.1, ...
    'learnRateDropPeriod', 5, ...
    'maxEpochs', 30, ...
    'miniBatchSize', 16, ...
    'shuffle', 'every-epoch', ...
    'validationFrequency', 30, ...
    'checkpointFrequency', 5, ...
    'checkpointFrequencyUnit', 'epoch');
end
